"""Local Streamlit interface for the file-based anomaly pipeline."""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from auth_anomaly.pipeline import analyze_file


st.set_page_config(page_title="Análisis de anomalías de autenticación", layout="wide")
st.title("Análisis de anomalías de autenticación")
st.write(
    "Analiza un archivo local de registros de autenticación y prioriza eventos para investigación. "
    "Las alertas no son una confirmación de actividad maliciosa."
)

with st.sidebar:
    st.header("Configuración")
    with st.form("analysis_settings"):
        input_path = st.text_input("Ruta local del archivo", value="data/rba/rba-dataset.csv")
        limit_rows = st.checkbox("Limitar filas procesadas", value=True)
        sample_rows = st.number_input(
            "Filas máximas",
            min_value=1,
            max_value=1_000_000,
            value=5_000,
            step=1_000,
            disabled=not limit_rows,
        )
        contamination = st.slider(
            "Contamination",
            min_value=0.005,
            max_value=0.49,
            value=0.02,
            step=0.005,
            help="Proporción aproximada que Isolation Forest marcará como anomalías; no es una tasa de aciertos.",
        )
        schema_map_text = st.text_area(
            "Mapeo opcional de roles (JSON)",
            placeholder='{"attack_label": "Is Account Takeover"}',
            help="Ejemplo: indica explícitamente la columna de etiqueta de referencia. La etiqueta no entrena el modelo.",
        )
        submitted = st.form_submit_button("Analizar archivo", type="primary", use_container_width=True)

if submitted:
    try:
        path = Path(input_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"No existe un archivo en la ruta indicada: {path}")
        schema_map: Dict[str, Any] = json.loads(schema_map_text) if schema_map_text.strip() else {}
        if not isinstance(schema_map, dict) or any(
            not isinstance(role, str) or not isinstance(column, str)
            for role, column in schema_map.items()
        ):
            raise ValueError("El mapeo debe ser un objeto JSON con roles y nombres de columna de tipo texto.")
        row_limit = int(sample_rows) if limit_rows else None

        with st.spinner("Procesando el archivo local..."):
            result = analyze_file(
                path,
                schema_map=schema_map,
                contamination=float(contamination),
                sample_rows=row_limit,
            )
        st.session_state["analysis_output"] = {
            "rows": result.scored_data,
            "report": result.report,
            "path": str(path),
            "sample_rows": row_limit,
            "contamination": float(contamination),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"No se pudo completar el análisis: {exc}")

output = st.session_state.get("analysis_output")
if output:
    report = output["report"]
    rows = output["rows"]
    evaluation = report["evaluation"]
    anomaly_count = int(report["anomaly_explanations"]["anomalies_found"])

    st.caption(
        f"Último análisis: `{output['path']}` · "
        f"límite de filas: {output['sample_rows'] or 'sin límite'} · "
        f"contamination: {output['contamination']:.3f}"
    )
    if evaluation["evaluation_type"] == "supervised":
        st.warning(
            "Las métricas supervisadas son in-sample: el modelo se ajustó y evaluó con estas mismas filas. "
            "No representan rendimiento en datos no vistos."
        )

    metrics = st.columns(4)
    metrics[0].metric("Filas leídas", f"{report['input_rows']:,}")
    metrics[1].metric("Filas después de limpieza", f"{report['clean_rows']:,}")
    metrics[2].metric("Anomalías marcadas", f"{anomaly_count:,}")
    metrics[3].metric("Tasa marcada", f"{evaluation['anomaly_rate']:.2%}")

    if evaluation["evaluation_type"] == "supervised":
        st.subheader("Métricas de referencia")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Precisión", f"{evaluation['precision']:.3f}")
        metric_columns[1].metric("Recall", f"{evaluation['recall']:.3f}")
        metric_columns[2].metric("F1", f"{evaluation['f1']:.3f}")
        metric_columns[3].metric("Tasa de falsos positivos", f"{evaluation['false_positive_rate']:.2%}")
    else:
        st.info("No se seleccionó una etiqueta binaria evaluable; se muestran estadísticas no supervisadas.")

    anomalies = rows.loc[rows["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    st.subheader("Eventos priorizados")
    st.caption("Se muestran hasta 100 alertas ordenadas por score. El score no es una probabilidad.")
    st.dataframe(anomalies.head(100), hide_index=True, use_container_width=True)
    st.download_button(
        "Descargar resultados CSV",
        data=rows.to_csv(index=False).encode("utf-8"),
        file_name="auth_anomaly_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    examples = report["anomaly_explanations"]["examples"]
    if examples:
        with st.expander("Contexto basado en reglas para alertas destacadas"):
            st.dataframe(pd.DataFrame(examples), hide_index=True, use_container_width=True)
            st.caption(report["anomaly_explanations"]["note"])

    with st.expander("Detalles del pipeline"):
        st.write("**Mapeo de roles**")
        st.json(report["schema_mapping"])
        st.write("**Variables generadas**")
        st.write(report["features"]["generated_features"])
        if report["features"]["omitted_features"]:
            st.write("**Variables omitidas**")
            st.write(report["features"]["omitted_features"])
        if report["warnings"]:
            st.write("**Advertencias**")
            for warning in report["warnings"]:
                st.warning(warning)
else:
    st.info("Indica una ruta local y presiona **Analizar archivo** para ejecutar el pipeline.")

st.caption(
    "Los datos se leen desde el entorno donde se ejecuta Streamlit. Para RBA, usa un archivo local "
    "y un límite de filas: el CSV completo es grande y no se carga desde el navegador."
)
