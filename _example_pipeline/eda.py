import skrub
import polars as pl
import plotly.express as px

import forecasting

out = forecasting.results_dir("eda")

for data_dir in forecasting.train_data_dir(), forecasting.test_data_dir():
    tables = [
        "<div><h2>{}</h2>{}</div>".format(
            t.stem,
            skrub.TableReport(
                forecasting.year_month_to_date(pl.read_csv(t))
            ).html_snippet(),
        )
        for t in data_dir.iterdir()
    ]

    html = f"""<!DOCTYPE html>
    <html>
    <head>
    <title>{data_dir.name}</title>
    </head>
    <body>
    <h1>{data_dir.name}</h1>
    {'\n'.join(tables)}
    </body>
    </html>
    """
    (out / f"{data_dir.name}.html").write_text(html, encoding="utf-8")


volume_hist = forecasting.year_month_to_date(
    pl.read_csv(forecasting.train_data_dir() / "historical_volume.csv")
)
for (agency,), df in volume_hist.group_by("Agency"):
    print(agency, end="\r")
    df = df.sort("YearMonth")
    fig = px.line(df, x="YearMonth", y="Volume", color="SKU", title=agency)
    fig.write_html(out / f"{agency}.html")
print()
