import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
	page_title="Machine Health Monitor",
	page_icon="M",
	layout="wide",
)


@st.cache_resource
def load_artifacts():
	model = joblib.load(BASE_DIR / "rf_model.joblib")
	scaler = joblib.load(BASE_DIR / "scaler.joblib")
	with open(BASE_DIR / "metadata_rfs.json", "r", encoding="utf-8") as metadata_file:
		metadata = json.load(metadata_file)
	return model, scaler, metadata


rf_model, scaler, metadata = load_artifacts()
numeric_cols = metadata["raw_numeric_cols"]
stats = metadata["feature_stats"]
failure_classes = list(rf_model.classes_)


def build_features(machine_type, inputs):
	values = [inputs[column] for column in numeric_cols]
	values.extend([int(machine_type == "L"), int(machine_type == "M")])
	return pd.DataFrame([values], columns=metadata["feature_cols"])


def analyze_machine(machine_type, inputs):
	features = build_features(machine_type, inputs)
	scaled_features = pd.DataFrame(
		scaler.transform(features), columns=metadata["feature_cols"]
	)
	probabilities = rf_model.predict_proba(scaled_features)[0]
	probability_by_class = dict(zip(failure_classes, probabilities))
	failure_probability = 1 - probability_by_class.get("No Failure", 0.0)
	quality_score = int(round((1 - failure_probability) * 100))
	return {
		"features": features,
		"prediction": rf_model.predict(scaled_features)[0],
		"probabilities": probability_by_class,
		"failure_probability": failure_probability,
		"quality_score": quality_score,
	}


def risk_band(failure_probability):
	if failure_probability < 0.2:
		return "Low", "success"
	if failure_probability < 0.5:
		return "Moderate", "warning"
	return "High", "error"


def operating_profile(features):
	rows = []
	for column in numeric_cols:
		column_stats = stats[column]
		value = float(features.iloc[0][column])
		z_score = (value - column_stats["mean"]) / column_stats["std"]
		rows.append({
			"Reading": column,
			"Value": value,
			"Z score": z_score,
			"Typical range": f"{column_stats['min']:.1f} - {column_stats['max']:.1f}",
		})
	return pd.DataFrame(rows)

st.title("Machine Health Monitor")
st.caption("An interactive decision cockpit for machine risk, operating discipline, and product quality.")

with st.sidebar:
	st.header("Live machine profile")
	st.caption("Tune the readings, then run one analysis across the model and statistics.")
	machine_type = st.selectbox("Machine type", metadata["type_categories"], index=0)
	inputs = {}
	with st.form("machine_form"):
		for column in numeric_cols:
			column_stats = stats[column]
			inputs[column] = st.number_input(
				column,
				min_value=float(column_stats["min"]),
				max_value=float(column_stats["max"]),
				value=float(column_stats["mean"]),
				step=0.1 if column != "Rotational speed [rpm]" else 1.0,
			)
		predict = st.form_submit_button("Analyze machine", type="primary", use_container_width=True)

if predict:
	st.session_state.analysis = analyze_machine(machine_type, inputs)
	st.session_state.analysis_inputs = inputs
	st.session_state.analysis_type = machine_type

if "analysis" in st.session_state:
	analysis = st.session_state.analysis
	prediction = analysis["prediction"]
	failure_probability = analysis["failure_probability"]
	quality_score = analysis["quality_score"]
	band, band_style = risk_band(failure_probability)

	st.subheader("Machine intelligence brief")
	result_columns = st.columns(4)
	result_columns[0].metric("Predicted status", "Healthy" if prediction == "No Failure" else "Failure risk")
	result_columns[1].metric("Quality score", f"{quality_score}/100")
	result_columns[2].metric("Failure probability", f"{failure_probability:.1%}")
	result_columns[3].metric("Risk band", band)

	if prediction == "No Failure":
		st.success("The current operating profile is predicted to remain stable.")
	else:
		if band_style == "error":
			st.error(f"The strongest predicted failure mode is {prediction}.")
		else:
			st.warning(f"The strongest predicted failure mode is {prediction}.")

	tab_overview, tab_diagnostics, tab_what_if = st.tabs(["Overview", "Statistical diagnostics", "What-if lab"])
	with tab_overview:
		left, right = st.columns([1.2, 1])
		with left:
			st.markdown("#### Failure probability by class")
			chart_data = pd.DataFrame(
				{"Probability": analysis["probabilities"]}
			).sort_values("Probability", ascending=False)
			st.bar_chart(chart_data, horizontal=True)
		with right:
			st.markdown("#### Model reasoning")
			importance = pd.Series(
				rf_model.feature_importances_, index=metadata["feature_cols"]
			).sort_values(ascending=False).head(5)
			st.dataframe(
				importance.rename("Importance").to_frame().style.format("{:.1%}"),
				use_container_width=True,
			)
			st.caption("Feature importance shows which inputs shaped the forest globally; it is not a causal explanation for one reading.")

	with tab_diagnostics:
		profile = operating_profile(analysis["features"])
		profile_display = profile.rename(columns={"Z score": "Standard deviations from mean"})
		st.markdown("#### How unusual is this machine profile?")
		st.dataframe(
			profile_display.style.format({"Value": "{:.1f}", "Standard deviations from mean": "{:+.2f}"}),
			use_container_width=True,
			hide_index=True,
		)
		st.caption("A value near 0 is close to the training average. Larger absolute values indicate a less typical operating condition.")
		st.markdown("#### Operating discipline")
		outlier_count = int((profile["Z score"].abs() > 2).sum())
		if outlier_count:
			st.error(f"{outlier_count} reading(s) are more than two standard deviations from the training mean.")
		else:
			st.success("All numeric readings are within two standard deviations of the training mean.")

	with tab_what_if:
		st.markdown("#### Compare a changed operating profile")
		st.caption("Adjust one lever to see whether the predicted risk moves. The model is recalculated immediately when you submit.")
		what_if_column, what_if_value = st.columns([1, 2])
		with what_if_column:
			what_if_feature = st.selectbox("Variable to change", numeric_cols, key="what_if_feature")
		with what_if_value:
			what_if_input = st.slider(
				f"New {what_if_feature}",
				min_value=float(stats[what_if_feature]["min"]),
				max_value=float(stats[what_if_feature]["max"]),
				value=float(st.session_state.analysis_inputs[what_if_feature]),
				step=0.1 if what_if_feature != "Rotational speed [rpm]" else 1.0,
			)
		what_if_inputs = dict(st.session_state.analysis_inputs)
		what_if_inputs[what_if_feature] = what_if_input
		what_if = analyze_machine(st.session_state.analysis_type, what_if_inputs)
		change_columns = st.columns(3)
		change_columns[0].metric("Current risk", f"{failure_probability:.1%}")
		change_columns[1].metric("What-if risk", f"{what_if['failure_probability']:.1%}", delta=f"{(what_if['failure_probability'] - failure_probability):+.1%}")
		change_columns[2].metric("What-if quality", f"{what_if['quality_score']}/100", delta=what_if["quality_score"] - quality_score)
		st.info(f"Changing {what_if_feature} to {what_if_input:.1f} produces: {what_if['prediction']}.")

	with st.expander("Show prepared model inputs"):
		st.dataframe(analysis["features"], use_container_width=True, hide_index=True)
else:
	st.info("Set the machine readings in the sidebar and select Analyze machine to unlock the intelligence brief.")

st.divider()
st.caption(
	"Quality score is derived from the model's probability of No Failure; it is not a separate quality model."
)
