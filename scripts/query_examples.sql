-- DERNIÈRES PRÉDICTIONS EFFECTUÉES
SELECT
    p.id         AS prediction_id,
    p.created_at AS prediction_date,
    p.model_version,
    p.predicted_class,
    p.predicted_proba,
    p.threshold_used,
    p.latency_ms,
    r.employee_id
FROM app.predictions p
JOIN app.prediction_requests r
  ON p.request_id = r.id
ORDER BY p.created_at DESC
LIMIT 20;


-- DERNIÈRES PRÉDICTIONS AVEC INFORMATIONS EMPLOYÉ
SELECT
    e.employee_external_id,
    e.departement,
    e.poste,
    p.predicted_class,
    p.predicted_proba,
    p.created_at
FROM app.predictions p
JOIN app.prediction_requests r
  ON p.request_id = r.id
JOIN raw.employees e
  ON r.employee_id = e.id
ORDER BY p.created_at DESC
LIMIT 20;


-- NOMBRE DE PRÉDICTIONS PAR CLASSE
SELECT
    predicted_class,
    COUNT(*) AS nb_predictions
FROM app.predictions
GROUP BY predicted_class
ORDER BY predicted_class;


-- TAUX DE RISQUE MOYEN PAR DÉPARTEMENT
SELECT
    e.departement,
    ROUND(AVG(p.predicted_proba)::numeric, 3) AS risque_moyen,
    COUNT(*) AS nb_predictions
FROM app.predictions p
JOIN app.prediction_requests r
  ON p.request_id = r.id
JOIN raw.employees e
  ON r.employee_id = e.id
GROUP BY e.departement
ORDER BY risque_moyen DESC;


-- EMPLOYÉS AVEC RISQUE ÉLEVÉ (> 0.8)
SELECT
    e.employee_external_id,
    e.departement,
    e.poste,
    p.predicted_proba,
    p.created_at
FROM app.predictions p
JOIN app.prediction_requests r
  ON p.request_id = r.id
JOIN raw.employees e
  ON r.employee_id = e.id
WHERE p.predicted_proba >= 0.8
ORDER BY p.predicted_proba DESC;


-- COMPARAISON PRÉDICTION vs RÉALITÉ (DERNIER GROUND TRUTH PAR EMPLOYÉ)
WITH last_truth AS (
  SELECT DISTINCT ON (employee_id)
    employee_id,
    a_quitte_l_entreprise,
    date_event
  FROM raw.ground_truth
  ORDER BY employee_id, date_event DESC
)
SELECT
    e.employee_external_id,
    p.predicted_class,
    gt.a_quitte_l_entreprise,
    p.predicted_proba,
    p.created_at  AS prediction_date,
    gt.date_event AS real_event_date
FROM app.predictions p
JOIN app.prediction_requests r
  ON p.request_id = r.id
JOIN raw.employees e
  ON r.employee_id = e.id
JOIN last_truth gt
  ON gt.employee_id = e.id
ORDER BY p.created_at DESC
LIMIT 50;


-- MATRICE DE CONFUSION (SIMPLIFIÉE) - DERNIER GROUND TRUTH PAR EMPLOYÉ
WITH last_truth AS (
  SELECT DISTINCT ON (employee_id)
    employee_id,
    a_quitte_l_entreprise
  FROM raw.ground_truth
  ORDER BY employee_id, date_event DESC
)
SELECT
    p.predicted_class          AS prediction,
    gt.a_quitte_l_entreprise   AS real_value,
    COUNT(*) AS count
FROM app.predictions p
JOIN app.prediction_requests r
  ON p.request_id = r.id
JOIN last_truth gt
  ON gt.employee_id = r.employee_id
GROUP BY p.predicted_class, gt.a_quitte_l_entreprise
ORDER BY p.predicted_class, gt.a_quitte_l_entreprise;


-- LATENCE MOYENNE DES PRÉDICTIONS
SELECT
    ROUND(AVG(latency_ms)::numeric, 2) AS avg_latency_ms,
    MAX(latency_ms)                    AS max_latency_ms,
    MIN(latency_ms)                    AS min_latency_ms
FROM app.predictions;


-- HISTORIQUE DES PRÉDICTIONS POUR UN EMPLOYÉ DONNÉ
SELECT
    p.created_at,
    p.predicted_class,
    p.predicted_proba,
    p.model_version
FROM app.predictions p
JOIN app.prediction_requests r
  ON p.request_id = r.id
JOIN raw.employees e
  ON r.employee_id = e.id
WHERE e.employee_external_id = 123
ORDER BY p.created_at DESC;


-- EMPLOYÉS À RISQUE MAIS TOUJOURS PRÉSENTS (DERNIER GROUND TRUTH)
WITH last_truth AS (
  SELECT DISTINCT ON (employee_id)
    employee_id,
    a_quitte_l_entreprise
  FROM raw.ground_truth
  ORDER BY employee_id, date_event DESC
)
SELECT
    e.employee_external_id,
    e.departement,
    e.poste,
    p.predicted_proba
FROM app.predictions p
JOIN app.prediction_requests r
  ON p.request_id = r.id
JOIN raw.employees e
  ON r.employee_id = e.id
JOIN last_truth gt
  ON gt.employee_id = e.id
WHERE p.predicted_class = 1
  AND gt.a_quitte_l_entreprise = 0
ORDER BY p.predicted_proba DESC;
