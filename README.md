<h1>Fire Risk Predictor – Predicting Time-to-Threat for Evacuation Zones Using Survival Analysis</h1>
<h3>Context</h3>
When a wildfire ignites, emergency responders face important questions with unfavourable circumstances:
<ul>
  <li>Which fires will reach populated areas?</li>
  <li>How quickly will those fires reach those areas?</li>
  <li>Which communities should prepare for the possibility of a wildfire reaching them first?</li>
</ul>
<h3>Problem Task</h3>
The task of this project is to build a survival model that answer these questions using only the earliest signals available. 
<ul style="font-size: 18px">
  <li>The model is to predict the probability that a wildfire will threaten an evacuation zone within 12, 24, 48, and 72 hours, drawing on data from just the first five hours after ignition.</li>
  <li>Emergency responders need both urgency rankings (which fires demand immediate attention) and probability estimates they can trust when making high-stakes decisions about evacuations, resource deployment, and public alerts.</li>
  <li>When a wildfire ignites, emergency managers and responders must decide which communities to warn, when to warn them, and where to position scarce resources.</li>
  <li>Many wildfire forecasting approaches reduce the task to a single question. Will this fire become dangerous. Emergency response needs more information because <b>decisions are time-bound and comparative</b>.</li>
</ul>
<h3>Dataset: WiDS Global Datathon 2026</h3>
<h4>Features</h4>
The dataset consisted of features pertaining to:
<ul>
  <li>Temporal coverage: e.g. <code>num_perimeters_0_5h</code>, <code>dt_first_last_0_5h</code></li>
  <li>Growth features: e.g. <code>log1p_area_first</code>, <code>area_first_ha</code>, <code>area_growth_abs_0_5h</code></li>
  <li>Centroid Kinematics: e.g. <code>centroid_displacement_m</code>, <code>spread_bearing_sin</code></li>
  <li>Distance to evacuation zone centroids: e.g. <code>dist_min_ci_0_5h</code>, <code>closing_speed_m_per_h</code></li>
  <li>Directionality: e.g. <code>alignment_abs</code></li>
  <li>Temporal metadata: e.g. <code>event_start_hour</code></li>
</ul>
<h4>Targets</h4>
<li><code>time_to_hit_hours</code>: Time from first five hours until fire comes within 5 km of an evac zone (hours). For censored events (never hit within 72h), this is the last observed time within the 72 hour observation window.</li>
<li><code>event</code>: Binary indicator where 1 if fire hits within 72 hours and 0 otherwise.</li>
<h4>Size</h4>
<ul>
  <li><code>train.csv</code>: 221 rows</li>
  <li><code>test.csv</code>: 95 rows</li>
</ul>
<h3>Modelling Approach</h3>
<h4>Model of choice</h4>
The <b>Cox Proportional Hazards</b> model was used for the following reasons:
<ul>
  <li>All features within the dataset, excluding <code>log1p_area_first</code> satisfied the <b>Proportional Hazards Assumption</b> necessary for the model to work.</li>
  <li><b>The training dataset was small in size</b> (221 rows) which meant that more advanced models such as <b>Random Survival Forest</b> would be <b>more susceptible to overfitting</b>.</li>
</ul>
<h4>Feature engineering</h4>
<ul>
  <li>Our best performing model utilised only <b>three</b> features from the dataset: <code>cross_track_component</code>, <code>num_perimeters_0_5h</code> and <code>dist_min_ci_0_5h</code>.</li>
  <li>An engineered feature using <code>num_perimeters_0_5h</code> and <code>dist_min_ci_0_5h</code> was used alongside <code>cross_track_component</code></li>
  <li>The engineered feature was <code>log_ratio_epsilonp</code> expressed as:</li>
  \[$\log(\epsilon + \frac{<code>num_perimeters_0_5h</code>}{<code>dist_min_ci_0_5h</code>})$\]
</ul>
