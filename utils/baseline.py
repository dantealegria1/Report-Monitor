import polars as pl
import math

def load_hourly_series_polars(
    csv_path: str,
    ts_col: str = "timestamp_hour",
    y_col: str = "y",
) -> pl.DataFrame:
    df = pl.read_csv(csv_path)

    if ts_col not in df.columns:
        raise ValueError(f"No existe la columna '{ts_col}' en el CSV.")
    if y_col not in df.columns:
        raise ValueError(f"No existe la columna '{y_col}' en el CSV.")

    # Parse timestamp y ordenar
    df = (
        df.with_columns(
            pl.col(ts_col)
            .str.strptime(pl.Datetime, strict=False)  # intenta parsear
            .alias(ts_col)
        )
        .sort(ts_col)
    )

    # Opcional: si tu CSV ya está por hora perfecto, no necesitas rellenar huecos.
    # Si quieres rellenar horas faltantes, usa un time-range y join (lo dejo simple por ahora).

    # Si hay nulls en y, los conviertes a 0
    df = df.with_columns(pl.col(y_col).fill_null(0))

    return df


def add_naive_predictions(df: pl.DataFrame, y_col: str = "y") -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(y_col).shift(1).alias("y_pred_last_hour"),
            pl.col(y_col).shift(168).alias("y_pred_last_week"),  # 24*7
        ]
    )


def _mae_rmse(df: pl.DataFrame, y_col: str, pred_col: str) -> tuple[float | None, float | None]:
    # Filtra filas válidas (sin null en pred)
    d = df.select([pl.col(y_col), pl.col(pred_col)]).drop_nulls()

    if d.height == 0:
        return None, None

    # MAE
    mae = (
        d.with_columns((pl.col(y_col) - pl.col(pred_col)).abs().alias("ae"))
        .select(pl.col("ae").mean())
        .item()
    )

    # RMSE
    mse = (
        d.with_columns(((pl.col(y_col) - pl.col(pred_col)) ** 2).alias("se"))
        .select(pl.col("se").mean())
        .item()
    )
    rmse = math.sqrt(mse) if mse is not None else None

    return float(mae), float(rmse)


def baseline_metrics(df: pl.DataFrame, y_col: str = "y") -> dict:
    mae1, rmse1 = _mae_rmse(df, y_col, "y_pred_last_hour")
    mae2, rmse2 = _mae_rmse(df, y_col, "y_pred_last_week")

    return {
        "MAE_last_hour": mae1,
        "RMSE_last_hour": rmse1,
        "MAE_last_week": mae2,   # None si no hay suficientes datos
        "RMSE_last_week": rmse2, # None si no hay suficientes datos
        "rows": df.height,
    }


def tail_window(df: pl.DataFrame, ts_col: str, hours: int) -> pl.DataFrame:
    # Asume 1 fila por hora; si no es así, te conviene filtrar por timestamp.
    return df.tail(hours)
