import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, abort, jsonify, session
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
from functools import wraps
from flask import Response
from scipy import stats
import holidays
from hijri_converter import convert
from config import Config
from models import db, User, Dataset, AnalysisResult

# ================= APP =================
app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

from flask_cors import CORS
CORS(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXT = ['.csv', '.xlsx']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOGIN =================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return wrap

# ================= DATA UTILS =================
def validate_columns_exist(df, columns):
    missing = [c for c in columns if c and c not in df.columns]
    return missing

def get_excel_sheets(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.xlsx':
        xls = pd.ExcelFile(path)
        return xls.sheet_names
    return []

def build_analysis_df(df, working_only):
    if not working_only:
        return df
    return df[
        (df['is_weekend'] == False) &
        (df['is_holiday'] == False)
    ]

def get_id_holidays(years):
    return holidays.Indonesia(years=years)

def enrich_calendar_flags(df, date_col):
    df = df.copy()
    years = df[date_col].dt.year.unique().tolist()
    ID_HOLIDAYS = get_id_holidays(years)

    df[date_col] = pd.to_datetime(df[date_col])
    df['date_only'] = df[date_col].dt.date
    # Weekend
    df['is_weekend'] = df[date_col].dt.weekday >= 5  # 5=Sabtu, 6=Minggu
    # Libur nasional
    df['is_holiday'] = df['date_only'].isin(ID_HOLIDAYS)
    df['holiday_name'] = df['date_only'].map(ID_HOLIDAYS).fillna('')
    # Ramadan
    def is_ramadan(d):
        try:
            h = convert.Gregorian(d.year, d.month, d.day).to_hijri()
            return h.month == 9
        except:
            return False
    df['is_ramadan'] = df['date_only'].apply(is_ramadan)
    def working_day(row):
        if row['is_holiday']:
            return False
        if row['is_weekend']:
            return False
        return True
    df['is_working_day'] = df.apply(working_day, axis=1)
    return df

def apply_working_day_filter(df, enabled=False, hide_holiday=False):
    if not enabled:
        return df
    df = df.copy()
    if hide_holiday:
        # MODE 2 → buang total
        df = df[df['is_working_day']]
    else:
        # MODE 1 → lewati libur tapi grafik nyambung
        df['working_index'] = (
            df['is_working_day']
            .astype(int)
            .cumsum()
        )
    return df

def calculate_pct_changes(values):
    pct = [None]
    for i in range(1, len(values)):
        prev = values[i-1]
        if prev == 0 or np.isnan(prev):
            pct.append(None)
        else:
            pct.append(((values[i] - prev) / prev) * 100)
    return pct

def flash_single(message, category='info'):
    session.pop('_flashes', None)
    flash(message, category)

def prepare_time_series(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col, ascending=True)
    return df.reset_index(drop=True)

def detect_date_columns(df, min_valid_ratio=0.7):
    date_cols = []
    for col in df.columns:
        try:
            # Skip numeric murni
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            valid_ratio = parsed.notna().mean()
            # Minimal variasi tanggal
            if valid_ratio >= min_valid_ratio and parsed.nunique() > 5:
                date_cols.append(col)
        except Exception:
            continue
    return date_cols

def get_active_date_column(df, dataset=None, user_selected=None):
    if dataset and dataset.date_column:
        return dataset.date_column, [dataset.date_column]
    date_cols = detect_date_columns(df)
    if user_selected and user_selected in date_cols:
        return user_selected, date_cols
    if len(date_cols) == 1:
        return date_cols[0], date_cols
    return None, date_cols

def load_dataset(path, sheet_name=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path)
    elif ext == '.xlsx':
        if sheet_name:
            return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        else:
            return pd.read_excel(path, engine="openpyxl")
    else:
        raise Exception("Format tidak didukung")

def load_active_dataset(dataset):
    print("DEBUG SHEET:", dataset.sheet_name)
    return load_dataset(
        dataset.path_file,
        sheet_name=dataset.sheet_name
    )

def resample_data(df, date_col, value_col, resample_freq='D', interpolate=True):
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        if interpolate and resample_freq and len(df) > 1:
            df_resampled = df.set_index(date_col)[value_col].resample(resample_freq).mean().reset_index().sort_values(date_col).reset_index(drop=True)
        else:
            df_resampled = df[[date_col, value_col]].copy()  # Fallback ke data asli
        return df_resampled
    except Exception as e:
        print(f"Error in resample_data: {e}")  # Debug
        return df[[date_col, value_col]].copy()  # Fallback
def detect_anomalies(values, threshold=1.5):
    values = np.array(values, dtype=float)
    if len(values) < 3:
        return np.zeros(len(values), dtype=bool)
    diffs = np.diff(values)
    std_diff = np.std(diffs)
    if std_diff == 0:
        return np.zeros(len(values), dtype=bool)
    anomalies = np.zeros(len(values), dtype=bool)
    anomalies[1:] = np.abs(diffs) > threshold * std_diff
    return anomalies
def classify_trend(values):
    values = np.array(values, dtype=float)
    if len(values) < 3:
        return "Sideways"
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    std = np.std(values)
    mean = np.mean(values)
    if std == 0 or mean == 0:
        return "Sideways"
    trend_strength = abs(slope) / std
    # ================= THRESHOLD =================
    STRONG_TREND = 0.002
    WEAK_TREND = 0.0008
    # ================= KLASIFIKASI =================
    if trend_strength >= STRONG_TREND:
        if slope > 0:
            return "Bullish"
        else:
            return "Bearish"
    elif trend_strength >= WEAK_TREND:
        if slope > 0:
            return "Sideways (Naik)"
        else:
            return "Sideways (Turun)"
    else:
        return "Sideways"

def plot_trend(df, date_col, value_col, use_resample=True):
    try:
        if use_resample:
            df_resampled = resample_data(df, date_col, value_col)
        else:
            df_resampled = resample_data(df, date_col, value_col, resample_freq=None, interpolate=False)
        if df_resampled.empty or len(df_resampled) < 2:
            raise ValueError("Data tidak cukup untuk plot")
        # Debug (hapus setelah fix)
        print("Data resampled shape:", df_resampled.shape)
        print("Value std:", df_resampled[value_col].std())
        # Detect anomalies using z-score
        y = df_resampled[value_col].values
        anomalies = detect_anomalies(y)
        diffs = np.diff(y)
        sharp_up = np.zeros(len(y), dtype=bool)
        sharp_down = np.zeros(len(y), dtype=bool)
        if len(diffs) > 1:
            std_diff = np.std(diffs)
            threshold = 1.5 * std_diff  
            for i in range(1, len(y)):
                delta = y[i] - y[i-1]
                if delta > threshold:
                    sharp_up[i] = True
                elif delta < -threshold:
                    sharp_down[i] = True
        # Calculate trend line
        df_resampled = df_resampled.sort_values(date_col).reset_index(drop=True)
        x_numeric = np.arange(len(df_resampled))
        slope, intercept = np.polyfit(x_numeric, df_resampled[value_col], 1)
        trend_line = slope * x_numeric + intercept
        fig = go.Figure()
        df_resampled[date_col] = ( pd.to_datetime(df_resampled[date_col], errors='coerce') .dt.strftime('%Y-%m-%d'))
        fig.add_trace(go.Scatter(x=df_resampled[date_col], y=df_resampled[value_col], mode='lines+markers', name='Data Asli', marker=dict(size=6, color='blue'), line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=df_resampled[date_col], y=trend_line, mode='lines', name='Trend Line', line=dict(color='orange', width=3, dash='dash')))
        if anomalies.any():
            fig.add_trace(go.Scatter(
                x=df_resampled[date_col][anomalies], 
                y=df_resampled[value_col][anomalies], 
                mode='markers', name='Anomali', 
                marker=dict(
                    color='red', 
                    size=10, 
                    symbol='x')
            ))
        # ================= NAIK TAJAM =================
        if sharp_up.any():
            fig.add_trace(go.Scatter(
                x=df_resampled[date_col][sharp_up],
                y=df_resampled[value_col][sharp_up],
                mode='markers',
                name='📈 Naik Tajam',
                marker=dict(
                    color='green',
                    size=14,
                    symbol='triangle-up',
                    line=dict(width=2, color='darkgreen')
                ),
                hovertemplate=
                    "<b style='color:green'>📈 NAIK TAJAM</b><br>" +
                    "Tanggal: %{x}<br>" +
                    f"{value_col}: %{y}<br>" +
                    "Perubahan signifikan<br>" +
                    "<extra></extra>"
            ))
        # ================= TURUN TAJAM =================
        if sharp_down.any():
            fig.add_trace(go.Scatter(
                x=df_resampled[date_col][sharp_down],
                y=df_resampled[value_col][sharp_down],
                mode='markers',
                name='📉 Turun Tajam',
                marker=dict(
                    color='purple',
                    size=14,
                    symbol='triangle-down',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate=
                    "<b style='color:red'>📉 TURUN TAJAM</b><br>" +
                    "Tanggal: %{x}<br>" +
                    f"{value_col}: %{y}<br>" +
                    "Penurunan signifikan<br>" +
                    "<extra></extra>"
            ))
        fig.update_layout(
            title=f"Analisis {value_col}",
            xaxis_title='Tanggal',
            yaxis_title=value_col,
            template="plotly_white",
            xaxis=dict(showgrid=True, autorange=True),
            yaxis=dict(showgrid=True, autorange=True),
            height=600,
            hovermode='x unified',
            showlegend=True,
            modebar=dict(
                add=['pan2d', 'select2d', 'lasso2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
            )
        )
        return fig.to_json(), anomalies
    except Exception as e:
        print(f"Error in plot_trend: {e}")  # Debug
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode='lines+markers', name='Data Asli'))
        fig.update_layout(title=f"Error: {str(e)} - Data Asli", xaxis_title='Tanggal', yaxis_title=value_col)
        return fig.to_json(), pd.Series([False] * len(df))

# ================= ANALYSIS =================
def analyze_and_save(df, dataset_id, column, date_col, working_only=False):
    df = df.copy()
    if 'is_working_day' not in df.columns:
        df = enrich_calendar_flags(df, date_col)
    df = df.sort_values(date_col).reset_index(drop=True)
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column]).reset_index(drop=True)
    # FILTER DATA
    analysis_df = build_analysis_df(df, working_only)
    if analysis_df.empty:
        return None
    values = analysis_df[column].values
    # STATISTIK
    mean = float(np.mean(values))
    median = float(np.median(values))
    std = float(np.std(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    count = len(values)
    anomalies = detect_anomalies(values)
    anomaly_count = int(anomalies.sum())
    trend_str = classify_trend(values)
    # DETAIL TREND
    increases, decreases = [], []
    increase_ranges, decrease_ranges = [], []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if diff > 0:
            increases.append(diff)
            increase_ranges.append((
                analysis_df[date_col].iloc[i - 1],
                analysis_df[date_col].iloc[i],
                float(diff)
            ))
        elif diff < 0:
            decreases.append(diff)
            decrease_ranges.append((
                analysis_df[date_col].iloc[i - 1],
                analysis_df[date_col].iloc[i],
                float(diff)
            ))
    pct_changes = [
        ((values[i] - values[i - 1]) / values[i - 1]) * 100
        for i in range(1, len(values))
        if values[i - 1] != 0
    ]
    # MODE ANALISIS
    analysis_mode = 'working_day' if working_only else 'full'
    analysis = AnalysisResult.query.filter_by(
        dataset_id=dataset_id,
        column_name=column,
        analysis_mode=analysis_mode
    ).first()
    if not analysis:
        analysis = AnalysisResult(
            dataset_id=dataset_id,
            column_name=column,
            analysis_mode=analysis_mode
        )
        db.session.add(analysis)
    analysis.mean = mean
    analysis.median = median
    analysis.std = std
    analysis.min = min_val
    analysis.max = max_val
    analysis.count = count
    analysis.anomaly_count = anomaly_count
    analysis.trend = trend_str
    db.session.commit()
    return {
        'mode': analysis_mode,
        'mean': mean,
        'median': median,
        'std': std,
        'min': min_val,
        'max': max_val,
        'count': count,
        'anomaly_count': anomaly_count,
        'trend': trend_str,
        'increases': increases,
        'decreases': decreases,
        'increase_ranges': increase_ranges,
        'decrease_ranges': decrease_ranges,
        'pct_changes': pct_changes,
        'total_increase': sum(increases) if increases else 0,
        'total_decrease': sum(decreases) if decreases else 0,
        'avg_pct_change': np.mean(pct_changes) if pct_changes else 0,
        'total_points': count
    }

def run_dual_analysis(df, dataset_id, column, date_col):
    if 'is_working_day' not in df.columns:
        df = enrich_calendar_flags(df, date_col)
    full_result = analyze_and_save(
        df,
        dataset_id,
        column,
        date_col,
        working_only=False
    )
    working_result = analyze_and_save(
        df,
        dataset_id,
        column,
        date_col,
        working_only=True
    )
    return {
        'full': full_result,
        'working_day': working_result
    }
# ================= ROUTES =================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('index'))
        flash_single("Login gagal")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    datasets = Dataset.query.filter_by(owner_id=current_user.id).all()
    dataset_id = request.args.get('dataset')
    active_dataset = None
    analysis_data = None
    plot_json = None
    numeric_cols = []
    selected_date_col = None
    date_cols = []
    selected_column = None
    sheets = []
    selected_sheet = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return redirect(url_for('index'))
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            flash_single("Format harus CSV / Excel")
            return redirect(url_for('index'))
        filename = secure_filename(file.filename)
        name = os.path.splitext(filename)[0]
        existing = [d.nama_dataset for d in datasets]
        base = name
        i = 1
        while name in existing:
            name = f"{base}({i})"
            i += 1
        save_name = f"user{current_user.id}_{name}{ext}"
        path = os.path.join(UPLOAD_FOLDER, save_name)
        file.save(path)
        df = load_dataset(path)
        dataset = Dataset(
            nama_dataset=name,
            filename=filename,
            path_file=path,
            total_rows=len(df),
            total_columns=len(df.columns),
            size_mb=round(os.path.getsize(path)/1024/1024, 2),
            owner_id=current_user.id
        )
        db.session.add(dataset)
        db.session.commit()
        return redirect(url_for('index', dataset=dataset.id))
    if dataset_id:
        active_dataset = Dataset.query.get_or_404(dataset_id)
        try:
            sheets = get_excel_sheets(active_dataset.path_file)
        except Exception as e:
            sheets = []
            flash(f"Dataset '{active_dataset.nama_dataset}' tidak valid: {str(e)}")
        # ====== SHEET DETECTION ======
        sheets = get_excel_sheets(active_dataset.path_file)
        selected_sheet = request.args.get('sheet')
        # LOGIKA SHEET
        if len(sheets) > 1:
            if not selected_sheet:
                # belum pilih sheet → stop di sini
                return render_template(
                    'index.html',
                    datasets=datasets,
                    active_dataset=active_dataset,
                    sheets=sheets,
                    selected_sheet=None,
                    date_cols=[],
                    numeric_cols=[],
                    analysis=None
                )
            df = load_dataset(active_dataset.path_file, sheet_name=selected_sheet)
            if selected_sheet != active_dataset.sheet_name:
                active_dataset.sheet_name = selected_sheet
                db.session.commit()
        else:
            # CSV atau Excel 1 sheet
            df = load_dataset(active_dataset.path_file)
            sheets = []
            selected_sheet = None
        selected_date_col = request.args.get('date_col')
        date_col, date_cols = get_active_date_column(
            df,
            active_dataset,
            selected_date_col
        )
        # ================= VALIDASI KOLUM TANGGAL =================
        missing = validate_columns_exist(df, [date_col])
        if missing:
            return render_template(
                'index.html',
                datasets=datasets,
                active_dataset=active_dataset,
                sheets=sheets,
                selected_sheet=selected_sheet,
                date_cols=date_cols,
                selected_date_col=selected_date_col,
                numeric_cols=[],
                analysis=None,
                plot_json=None,
                error_message=(
                    f"Kolom waktu '{missing[0]}' tidak tersedia di sheet ini. "
                    "Mohon untuk ganti kembali sheetnya."
                )
            )
        if not date_col:
            flash(f"Pilih kolom waktu terlebih dahulu: {', '.join(date_cols)}")
            return render_template(
                'index.html',
                datasets=datasets,
                active_dataset=active_dataset,
                date_cols=date_cols,
                selected_date_col=selected_date_col,
                numeric_cols=[],
                analysis=None
            )
        working_only = request.args.get('working_day') == '1'
        hide_holiday = request.args.get('hide_holiday') == '1'
        df = prepare_time_series(df, date_col)
        df = enrich_calendar_flags(df, date_col)
        df_full = df.copy()
        analysis_df = build_analysis_df(df_full, working_only)
        if selected_date_col and selected_date_col != active_dataset.date_column:
            active_dataset.date_column = selected_date_col
            db.session.commit()
        # ================= GRAPH DF (UNTUK VISUAL) =================
        df_plot = df_full.copy()
        numeric_cols = [
            c for c in analysis_df.select_dtypes(include='number').columns
            if c not in ['is_weekend', 'is_holiday', 'is_working_day', 'working_index']
        ]
        selected_column = request.args.get('column') or (
            numeric_cols[0] if numeric_cols else None
        )
        if selected_column:
            analysis_data = analyze_and_save(
                df_full,
                active_dataset.id,
                selected_column,
                date_col,
                working_only=working_only
            )
            # ====== BUAT PLOT ======
            plot_json, _ = plot_trend(
                df_plot,
                date_col,
                selected_column,
                use_resample=False
            )
        else:
            analysis_data = None
            plot_json = None     
    return render_template(
        'index.html',
        datasets=datasets,
        active_dataset=active_dataset,
        sheets=sheets,
        selected_sheet=selected_sheet,
        analysis=analysis_data,
        analysis_data=analysis_data,
        plot_json=plot_json,
        numeric_cols=numeric_cols,
        selected_column=selected_column,
        date_cols=date_cols if 'date_cols' in locals() else [],
        selected_date_col=selected_date_col
    )

# ================= API ENDPOINTS =================
@app.route('/api/yearly-yoy/<int:dataset_id>/<int:year>')
@login_required
def api_yearly_yoy(dataset_id, year):
    column = request.args.get('column')
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    date_col, _ = get_active_date_column(df, dataset, request.args.get('date_col'))
    missing = validate_columns_exist(df, [date_col, column])
    if missing:
        return jsonify({
            "status": "error",
            "type": "MISSING_COLUMN",
            "message": (
                f"Kolom '{missing[0]}' tidak tersedia di sheet ini. "
                "Mohon untuk ganti kembali sheetnya."
            ),
            "missing_column": missing[0]
        }), 200
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[date_col, column])
    # tahun sekarang & sebelumnya
    cur = df[df[date_col].dt.year == year]
    prev = df[df[date_col].dt.year == year - 1]
    cur_month = cur.groupby(cur[date_col].dt.month)[column].sum()
    prev_month = prev.groupby(prev[date_col].dt.month)[column].sum()
    yoy = []
    for m in cur_month.index:
        if m in prev_month and prev_month[m] != 0:
            yoy.append(((cur_month[m] - prev_month[m]) / prev_month[m]) * 100)
        else:
            yoy.append(None)
    return jsonify({
        "months": cur_month.index.tolist(),
        "values": cur_month.values.tolist(),
        "yoy": [None if v is None else round(v, 2) for v in yoy]
    })

@app.route('/api/monthly-mom/<int:dataset_id>/<int:year>/<int:month>')
@login_required
def api_monthly_mom(dataset_id, year, month):
    column = request.args.get('column')
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    date_col, _ = get_active_date_column(df, dataset, request.args.get('date_col'))
    missing = validate_columns_exist(df, [date_col, column])
    if missing:
        return jsonify({
            "status": "error",
            "type": "MISSING_COLUMN",
            "message": (
                f"Kolom '{missing[0]}' tidak tersedia di sheet ini. "
                "Mohon untuk ganti kembali sheetnya."
            ),
            "missing_column": missing[0]
        }), 200
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[date_col, column])
    cur = df[(df[date_col].dt.year == year) & (df[date_col].dt.month == month)]
    prev = df[(df[date_col].dt.year == year) & (df[date_col].dt.month == month - 1)]
    cur_sum = cur[column].sum()
    prev_sum = prev[column].sum()
    mom = None
    if prev_sum != 0:
        mom = ((cur_sum - prev_sum) / prev_sum) * 100
    return jsonify({
        "value": float(cur_sum),
        "mom": None if mom is None else round(mom, 2)
    })

@app.route('/api/yearly_with_monthly/<int:dataset_id>/<int:year>')
@login_required
def api_yearly_with_monthly(dataset_id, year):
    column = request.args.get('column')
    if not column:
        return jsonify({"error": "column required"}), 400
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    # cari kolom tanggal fleksibel
    user_date_col = request.args.get('date_col')
    date_col, _ = get_active_date_column(
        df,
        dataset,
        user_date_col
    )
    missing = validate_columns_exist(df, [date_col, column])
    if missing:
        return jsonify({
            "status": "error",
            "type": "MISSING_COLUMN",
            "message": (
                f"Kolom '{missing[0]}' tidak tersedia di sheet ini. "
                "Mohon untuk ganti kembali sheetnya."
            ),
            "missing_column": missing[0]
        }), 200
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[date_col, column])
    df = prepare_time_series(df, date_col)
    df = enrich_calendar_flags(df, date_col)
    working_only = request.args.get('working_only') == '1'
    hide_holiday = request.args.get('hide_holiday') == '1'
    analysis_df = build_analysis_df(df, working_only)
    analysis_df = analysis_df[
        analysis_df[date_col].dt.year == year
    ]
    df = df[df[date_col].dt.year == year]
    if df.empty:
        return jsonify({"data": [], "trend_line": [], "anomalies": [], "trend": "Sideways"})
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['days_in_month'] = df[date_col].dt.days_in_month
    monthly = []
    for m, g in df.groupby('month'):
        # resample harian untuk bulan tidak penuh
        g_resampled = g.set_index(date_col)[column].resample('D').mean().interpolate().reset_index()
        monthly_sum = float(g_resampled[column].sum()) if not g_resampled.empty else 0
        monthly.append({
            "month": int(m),
            "value": monthly_sum,
            "days_count": len(g_resampled),
            "is_full": g['day'].min() == 1 and g['day'].max() == g['days_in_month'].iloc[0]
        })
    monthly = sorted(monthly, key=lambda x: x['month'])
    values = np.array([m['value'] for m in monthly])
    pct_changes = calculate_pct_changes(values)
    # TREND & ANOMALI
    anomalies = detect_anomalies(values).tolist()
    x_numeric = np.arange(len(values))
    slope, intercept = np.polyfit(x_numeric, values, 1) if len(values) > 1 else (0, values[0] if len(values) else 0)
    trend_line = (slope * x_numeric + intercept).tolist()
    trend_str = classify_trend(values)
    return jsonify({
        "data": monthly,
        "trend_line": trend_line,
        "anomalies": anomalies,
        "trend": trend_str,
        "pct_changes": [
            None if p is None else round(float(p), 2)
            for p in pct_changes
        ]
    })

@app.route('/api/periods/<int:dataset_id>')
@login_required
def api_periods(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    user_date_col = request.args.get('date_col')
    date_col, _ = get_active_date_column(df, dataset, user_date_col)
    if not date_col:
        return jsonify({})
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['days_in_month'] = df[date_col].dt.days_in_month
    periods = {}
    for (y, m), g in df.groupby(['year', 'month']):
        is_full = (
            g['day'].min() == 1 and
            g['day'].max() == g['days_in_month'].iloc[0]
        )
        periods.setdefault(str(y), []).append({
            "month": int(m),
            "month_name": pd.to_datetime(f"{int(y)}-{int(m)}-01").strftime("%B"),
            "is_full": bool(is_full)
        })
    return jsonify(periods)

@app.route('/api/yearly-analysis/<int:dataset_id>/<int:year>')
@login_required
def api_yearly_analysis(dataset_id, year):
    column = request.args.get('column')
    if not column:
        return jsonify({"error": "column required"}), 400
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    user_date_col = request.args.get('date_col')
    date_col, _ = get_active_date_column(
        df,
        dataset,
        user_date_col
    )
    missing = validate_columns_exist(df, [date_col, column])
    if missing:
        return jsonify({
            "status": "error",
            "type": "MISSING_COLUMN",
            "message": (
                f"Kolom '{missing[0]}' tidak tersedia di sheet ini. "
                "Mohon untuk ganti kembali sheetnya."
            ),
            "missing_column": missing[0]
        }), 200
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[date_col, column])
    df = prepare_time_series(df, date_col)
    df = enrich_calendar_flags(df, date_col)
    working_only = request.args.get('working_only') == '1'
    hide_holiday = request.args.get('hide_holiday') == '1'
    analysis_df = build_analysis_df(df, working_only)
    analysis_df = analysis_df[
        analysis_df[date_col].dt.year == year
    ]
    df = df[df[date_col].dt.year == year]
    if df.empty:
        return jsonify({"error": "no data"}), 404
    analysis_df['month'] = analysis_df[date_col].dt.month
    monthly = analysis_df.groupby('month')[column].sum().sort_index()
    values = monthly.values
    anomalies = detect_anomalies(values)
    anomaly_count = int(anomalies.sum())
    diffs = np.diff(values)
    std_diff = np.std(diffs) if len(diffs) > 1 else 0
    threshold = 1.5 * std_diff
    sharp_up = np.zeros(len(values), dtype=bool)
    sharp_down = np.zeros(len(values), dtype=bool)
    for i in range(1, len(values)):
        delta = values[i] - values[i-1]
        if delta > threshold:
            sharp_up[i] = True
        elif delta < -threshold:
            sharp_down[i] = True
    diffs = np.diff(values)
    increases = diffs[diffs > 0]
    decreases = diffs[diffs < 0]
    pct = (diffs / values[:-1]) * 100 if len(values) > 1 else []
    return jsonify({
    "mean": round(float(np.mean(values)), 2),
    "median": round(float(np.median(values)), 2),
    "std": round(float(np.std(values)), 2),
    "trend": classify_trend(values),
    "anomaly_count": anomaly_count,
    "total_increase": round(float(increases.sum()), 2) if len(increases) else 0,
    "total_decrease": round(float(decreases.sum()), 2) if len(decreases) else 0,
    "increase_count": int(len(increases)),
    "decrease_count": int(len(decreases)),
    "avg_pct": round(float(np.mean(pct)), 2) if len(pct) else 0,
    "max_pct": round(float(np.max(pct)), 2) if len(pct) else 0,
    "min_pct": round(float(np.min(pct)), 2) if len(pct) else 0,
    "valid_months": len(values),
    "completeness_pct": round((len(values) / 12) * 100, 1),
    "sharp_up": sharp_up.tolist(),
    "sharp_down": sharp_down.tolist(),
})

@app.route('/api/monthly-analysis/<int:dataset_id>/<int:year>/<int:month>')
@login_required
def api_monthly_analysis(dataset_id, year, month):
    column = request.args.get('column')
    if not column:
        return jsonify({"error": "column required"}), 400
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    # Temukan kolom tanggal
    user_date_col = request.args.get('date_col')
    date_col, _ = get_active_date_column(
        df,
        dataset,
        user_date_col
    )
    missing = validate_columns_exist(df, [date_col, column])
    if missing:
        return jsonify({
            "status": "error",
            "type": "MISSING_COLUMN",
            "message": (
                f"Kolom '{missing[0]}' tidak tersedia di sheet ini. "
                "Mohon untuk ganti kembali sheetnya."
            ),
            "missing_column": missing[0]
        }), 200
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[date_col, column])
    # Filter bulan & tahun
    df = df[(df[date_col].dt.year == year) & (df[date_col].dt.month == month)].sort_values(date_col)
    if df.empty:
        return jsonify({"error": "no data"}), 404
    df = enrich_calendar_flags(df, date_col)
    working_only = request.args.get('working_only') == '1'
    analysis_df = build_analysis_df(df, working_only)
    df = analysis_df[
        (analysis_df[date_col].dt.year == year) &
        (analysis_df[date_col].dt.month == month)
    ]
    # Resample harian agar tanggal lengkap
    df_resampled = df.set_index(date_col)[column].resample('D').mean().interpolate().reset_index()
    y = df_resampled[column].values
    x_dates = df_resampled[date_col].dt.strftime('%Y-%m-%d').tolist()
    # Trend line
    x_num = np.arange(len(y))
    slope, intercept = np.polyfit(x_num, y, 1)
    trend_line = (slope * x_num + intercept).tolist()
    # Anomalies
    anomalies = detect_anomalies(y)
    # Sharp up/down detection (mirip plot utama)
    diffs = np.diff(y)
    increases = diffs[diffs > 0]
    decreases = diffs[diffs < 0]
    total_increase = float(increases.sum()) if len(increases) else 0
    total_decrease = float(decreases.sum()) if len(decreases) else 0
    increase_count = int(len(increases))
    decrease_count = int(len(decreases))
    std_diff = np.std(diffs) if len(diffs) > 1 else 0
    threshold = 1.5 * std_diff
    sharp_up = np.zeros(len(y), dtype=bool)
    sharp_down = np.zeros(len(y), dtype=bool)
    for i in range(1, len(y)):
        delta = y[i] - y[i-1]
        if delta > threshold:
            sharp_up[i] = True
        elif delta < -threshold:
            sharp_down[i] = True
    # Persentase perubahan harian
    pct_changes = [((y[i]-y[i-1])/y[i-1])*100 for i in range(1,len(y)) if y[i-1] != 0]
    # Kelengkapan & metrik
    missing_count = df_resampled[column].isna().sum()
    completeness_pct = (len(y) - missing_count) / len(y) * 100
    return jsonify({
    "mean": float(np.mean(y)),
    "median": float(np.median(y)),
    "std": float(np.std(y)),
    "anomaly_count": int(anomalies.sum()),
    "trend": classify_trend(y),
    "total_increase": round(total_increase, 2),
    "total_decrease": round(total_decrease, 2),
    "increase_count": increase_count,
    "decrease_count": decrease_count,
    "avg_pct": round(float(np.mean(pct_changes)), 2) if pct_changes else 0,
    "max_pct": float(np.max(pct_changes)) if pct_changes else 0,
    "min_pct": float(np.min(pct_changes)) if pct_changes else 0,
    "missing_count": int(missing_count),
    "completeness_pct": round(completeness_pct, 1),
    "valid_days": len(y),
    "total_points": len(y),
    "x": x_dates,
    "y": y.tolist(),
    "trend_line": trend_line,
    "anomalies": anomalies.tolist(),
})

@app.route('/api/yearly/<int:dataset_id>/<int:year>')
@login_required
def api_yearly(dataset_id, year):
    column = request.args.get('column')
    if not column:
        return jsonify({"error": "column required"}), 400
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    # Flexible date column
    user_date_col = request.args.get('date_col')
    date_col, _ = get_active_date_column(
        df,
        dataset,
        user_date_col
    )
    missing = validate_columns_exist(df, [date_col, column])
    if missing:
        return jsonify({
            "status": "error",
            "type": "MISSING_COLUMN",
            "message": (
                f"Kolom '{missing[0]}' tidak tersedia di sheet ini. "
                "Mohon untuk ganti kembali sheetnya."
            ),
            "missing_column": missing[0]
        }), 200
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[date_col, column])
    df = prepare_time_series(df, date_col)
    df = enrich_calendar_flags(df, date_col)
    working_only = request.args.get('working_only') == '1'
    hide_holiday = request.args.get('hide_holiday') == '1'
    analysis_df = build_analysis_df(df, working_only)
    analysis_df = analysis_df[
        analysis_df[date_col].dt.year == year
    ]
    df = df[df[date_col].dt.year == year]
    if df.empty:
        return jsonify({"data": [], "trend_line": [], "anomalies": [], "trend": "Sideways"})
    # Hitung total per bulan tanpa skip
    monthly = analysis_df.groupby(
        analysis_df[date_col].dt.month
    )[column].sum().sort_index()
    values = monthly.values
    months = monthly.index.tolist()
    pct_changes = calculate_pct_changes(values)
    # TREND & ANOMALI
    anomalies = detect_anomalies(values).tolist()
    x_numeric = np.arange(len(values))
    slope, intercept = np.polyfit(x_numeric, values, 1) if len(values) > 1 else (0, values[0])
    trend_line = (slope * x_numeric + intercept).tolist()
    trend_str = classify_trend(values)
    # Format data bulanan
    monthly_data = [{"month": int(m), "value": float(v)} for m, v in zip(months, values)]
    return jsonify({
        "data": monthly_data,
        "trend_line": trend_line,
        "anomalies": anomalies,
        "trend": trend_str,
        "pct_changes": [
            None if p is None else round(float(p), 2)
            for p in pct_changes
        ]
    })

@app.route('/api/monthly/<int:dataset_id>/<int:year>/<int:month>')
@login_required
def api_monthly(dataset_id, year, month):
    column = request.args.get('column')
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    user_date_col = request.args.get('date_col')
    date_col, _ = get_active_date_column(
        df,
        dataset,
        user_date_col
    )
    missing = validate_columns_exist(df, [date_col, column])
    if missing:
        return jsonify({
            "status": "error",
            "type": "MISSING_COLUMN",
            "message": (
                f"Kolom '{missing[0]}' tidak tersedia di sheet ini. "
                "Mohon untuk ganti kembali sheetnya."
            ),
            "missing_column": missing[0]
        }), 200
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[date_col, column])
    df = df[(df[date_col].dt.year == year) & (df[date_col].dt.month == month)].sort_values(date_col)
    # Enrich calendar flags so frontend can render holiday/weekend/ramadan shading
    df = enrich_calendar_flags(df, date_col)
    # Respect working day / hide non-working query params coming from frontend
    working_only = request.args.get('working_day') == '1'
    hide_holiday = request.args.get('hide_non_working') == '1'
    # Build dataframe used for plotting (may be filtered when hiding non-working days)
    if working_only and hide_holiday:
        df_plot = df[df['is_working_day']].copy()
    else:
        df_plot = apply_working_day_filter(df, enabled=working_only, hide_holiday=hide_holiday)
    if df.empty:
        return jsonify({"x": [], "y": [], "trend_line": [], "anomalies": []})
    y = df_plot[column].values
    pct_changes = calculate_pct_changes(y)
    x_dates = df_plot[date_col].dt.strftime('%Y-%m-%d').tolist()
    anomalies = detect_anomalies(y).tolist()
    x_num = np.arange(len(y))
    slope, intercept = np.polyfit(x_num, y, 1) if len(y) > 1 else (0, float(y[0]) if len(y) else 0)
    trend_line = (slope * x_num + intercept).tolist() if len(y) else []
    # Build calendar metadata (always use date-based x so monthly shading aligns)
    calendar = []
    for i, row in df.iterrows():
        calendar.append({
            "x": row[date_col].strftime('%Y-%m-%d'),
            "date": row[date_col].strftime('%Y-%m-%d'),
            "is_weekend": bool(row['is_weekend']),
            "is_holiday": bool(row['is_holiday']),
            "is_ramadan": bool(row['is_ramadan']),
            "holiday_name": row['holiday_name'] or ""
        })
    return jsonify({
        "x": x_dates,
        "y": y.tolist(),
        "trend_line": trend_line,
        "anomalies": anomalies,
        "trend": classify_trend(y),
        "pct_changes": [
            None if p is None else round(float(p), 2)
            for p in pct_changes
        ],
        "calendar": calendar
    })

@app.route('/api/user')
@login_required
def api_user():
    return jsonify({
        'username': current_user.username,
        'role': current_user.role
    })

@app.route('/api/datasets')
@login_required
def api_datasets():
    datasets = Dataset.query.filter_by(owner_id=current_user.id).all()
    return jsonify([{
        'id': d.id,
        'nama_dataset': d.nama_dataset
    } for d in datasets])

@app.route('/api/dataset/<int:dataset_id>')
@login_required
def api_dataset_detail(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.owner_id != current_user.id and current_user.role != 'admin':
        abort(403)
    df = load_active_dataset(dataset)
    user_date_col = request.args.get('date_col')
    date_col, date_cols = get_active_date_column(df, user_date_col)
    if not date_col:
        return jsonify({
            'error': f"Pilih kolom waktu terlebih dahulu: {date_cols}"
        }), 400
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        return jsonify({'error': 'Tidak ada kolom numerik'}), 400
    column = numeric_cols[0]
    analysis_data = analyze_and_save(df, dataset.id, column, date_col)
    plot_json, _ = plot_trend(df, date_col, column, use_resample=False)
    return jsonify({
        'nama_dataset': dataset.nama_dataset,
        'numeric_cols': numeric_cols,
        'analysis': {
            'mean': analysis_data['analysis'].mean,
            'median': analysis_data['analysis'].median,
            'std': analysis_data['analysis'].std,
            'anomaly_count': analysis_data['analysis'].anomaly_count,
            'trend': analysis_data['analysis'].trend
        },
        'analysis_data': {
            'total_increase': analysis_data['total_increase'],
            'increases': analysis_data['increases'],
            'total_decrease': analysis_data['total_decrease'],
            'decreases': analysis_data['decreases'],
            'avg_pct_change': analysis_data['avg_pct_change'],
            'pct_changes': analysis_data['pct_changes'],
            'completeness_pct': analysis_data['completeness_pct'],
            'missing_count': analysis_data['missing_count']
        },
        'count': analysis_data['analysis'].count,
        'plot_json': plot_json
    })
    
@app.route('/stats/<int:dataset_id>')
@login_required
def get_stats(dataset_id):
    column = request.args.get('column')
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    user_date_col = request.args.get('date_col')
    date_col, _ = get_active_date_column(
        df,
        dataset,
        user_date_col
    )
    missing = validate_columns_exist(df, [date_col, column])
    if missing:
        return jsonify({
            "status": "error",
            "type": "MISSING_COLUMN",
            "message": (
                f"Kolom '{missing[0]}' tidak tersedia di sheet ini. "
                "Mohon untuk ganti kembali sheetnya."
            ),
            "missing_column": missing[0]
        }), 200
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[date_col, column]).reset_index(drop=True)
    df = enrich_calendar_flags(df, date_col)
    working_only = request.args.get('working_only') == '1'
    analysis_df = build_analysis_df(df, working_only)
    values = analysis_df[column].values
    mean = float(np.mean(values))
    median = float(np.median(values))
    std = float(np.std(values))
    count = len(values)
    # Trend (slope)
    trend = classify_trend(values)
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    std = np.std(values)
    trend_strength = abs(slope) / std if std != 0 else 0
    # Anomali (z-score)
    anomalies = detect_anomalies(values)
    anomaly_indexes = np.where(anomalies)[0].tolist()
    anomaly_count = int(anomalies.sum())
    # Perubahan
    diffs = np.diff(values)
    std_diff = np.std(diffs) if len(diffs) > 1 else 0
    threshold = 1.5 * std_diff
    sharp_up = np.zeros(len(values), dtype=bool)
    sharp_down = np.zeros(len(values), dtype=bool)
    for i in range(1, len(values)):
        delta = values[i] - values[i-1]
        if delta > threshold:
            sharp_up[i] = True
        elif delta < -threshold:
            sharp_down[i] = True
    diffs = np.diff(values)
    increases = diffs[diffs > 0]
    decreases = diffs[diffs < 0]
    pct_changes = np.diff(values) / values[:-1] * 100 if len(values) > 1 else []
    return jsonify({
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "count": count,
        "trend": trend,
        "anomaly_count": anomaly_count,
        "anomaly_indexes": anomaly_indexes,
        "trend_strength": round(trend_strength, 3),
        "total_increase": float(increases.sum()) if len(increases) else 0,
        "total_decrease": float(decreases.sum()) if len(decreases) else 0,
        "increase_count": int(len(increases)),
        "decrease_count": int(len(decreases)),
        "avg_pct_change": float(np.mean(pct_changes)) if len(pct_changes) else 0,
        "max_pct": float(np.max(pct_changes)) if len(pct_changes) else 0,
        "min_pct": float(np.min(pct_changes)) if len(pct_changes) else 0,
        "missing_count": int(df[column].isna().sum()),
        "completeness_pct": round((count / len(df)) * 100, 1)
    })

# ================= AJAX PLOT =================
@app.route('/plot/<int:dataset_id>')
@login_required
def get_plot(dataset_id):
    column = request.args.get('column')
    resample_str = request.args.get('resample', 'false')
    use_resample = resample_str.lower() in ['true', '1', 'yes']
    dataset = Dataset.query.get_or_404(dataset_id)
    df = load_active_dataset(dataset)
    user_date_col = request.args.get('date_col')
    date_col, _ = get_active_date_column(
        df,
        dataset,
        user_date_col
    )
    missing = validate_columns_exist(df, [date_col, column])
    if missing:
        return jsonify({
            "status": "error",
            "type": "MISSING_COLUMN",
            "message": (
                f"Kolom '{missing[0]}' tidak tersedia di sheet ini. "
                "Mohon untuk ganti kembali sheetnya."
            ),
            "missing_column": missing[0]
        }), 200
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = prepare_time_series(df, date_col)
    df = enrich_calendar_flags(df, date_col)
    working_only = request.args.get('working_only') == '1'
    hide_holiday = request.args.get('hide_holiday') == '1'
    if working_only:
        df = df[df['is_working_day']]
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])
    if use_resample:
        df = df.set_index(date_col)[column] \
            .resample('D').mean().interpolate() \
            .reset_index()

        df = enrich_calendar_flags(df, date_col)
        pass
    y = df[column].values
    if working_only and not hide_holiday:
        x = df['working_index'].tolist()
    else:
        x = df[date_col].dt.strftime('%Y-%m-%d').tolist()
    pct_change = [None]
    for i in range(1, len(y)):
        if y[i-1] != 0:
            pct_change.append(((y[i] - y[i-1]) / y[i-1]) * 100)
        else:
            pct_change.append(0)
    # Trend line
    x_num = np.arange(len(y))
    slope, intercept = np.polyfit(x_num, y, 1)
    trend_line = (slope * x_num + intercept).tolist()
    # Anomalies
    anomalies = detect_anomalies(y).tolist()
    # Sharp up/down detection
    diffs = np.diff(y)
    std_diff = np.std(diffs) if len(diffs) > 1 else 0
    threshold = 1.5 * std_diff
    sharp_up = np.array([False]*len(y))
    sharp_down = np.array([False]*len(y))
    for i in range(1, len(y)):
        delta = y[i] - y[i-1]
        if delta > threshold:
            sharp_up[i] = True
        elif delta < -threshold:
            sharp_down[i] = True
    calendar = []
    for i, row in df.iterrows():
        calendar.append({
            "x": row['working_index']
                if working_only and not hide_holiday
                else row[date_col].strftime('%Y-%m-%d'),
            "date": row[date_col].strftime('%Y-%m-%d'),
            "is_weekend": bool(row['is_weekend']),
            "is_holiday": bool(row['is_holiday']),
            "is_ramadan": bool(row['is_ramadan']),
            "holiday_name": row['holiday_name'] or ""
        })
    return jsonify({
        "x": x,
        "y": y.tolist(),
        "pct_change": [
            round(p, 2) if p is not None else None
            for p in pct_change
        ],
        "trend": trend_line,
        "anomalies": anomalies,
        "sharp_up": sharp_up.tolist(),
        "sharp_down": sharp_down.tolist(),
        "meta": {
            "min": float(min(y)),
            "max": float(max(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "slope": slope
        },
        "calendar": calendar
    })

# ================= DELETE DATASET =================
@app.route('/dataset/delete/<int:dataset_id>', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.owner_id != current_user.id and current_user.role != 'admin':
        abort(403)
    AnalysisResult.query.filter_by(dataset_id=dataset.id).delete()
    db.session.commit()
    if os.path.exists(dataset.path_file):
        os.remove(dataset.path_file)
    db.session.delete(dataset)
    db.session.commit()
    flash_single("Dataset berhasil dihapus")
    return redirect(url_for('index'))

# ================= ADMIN =================
@app.route('/admin/users', methods=['GET', 'POST'])
@login_required
@admin_required
def manage_users():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        if User.query.filter_by(username=username).first():
            flash_single("Username sudah ada")
            return redirect(url_for('manage_users'))
        user = User(
            username=username,
            password=generate_password_hash(password),
            role=role
        )
        db.session.add(user)
        db.session.commit()
        flash_single("User ditambahkan")
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash_single("Tidak bisa hapus diri sendiri")
    else:
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for('manage_users'))

# ================= INIT =================
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        user = User(username='admin', password=generate_password_hash('admin'), role='admin')
        db.session.add(user)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True)