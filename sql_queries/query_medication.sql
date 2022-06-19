SELECT * FROM prescriptions 
    INNER JOIN
    (
        SELECT * FROM (
            (SELECT * FROM labevents) AS subquery2 
            INNER JOIN 
            (SELECT * FROM d_labitems) AS subquery3 
            ON subquery2.itemid=subquery3.itemid
        ) AS subquery1 
        WHERE 
        subquery1.hadm_id IS NOT NULL 
        AND
        (
            subquery1.hadm_id IN (
                SELECT DISTINCT ON (subject_id) hadm_id FROM admissions
                WHERE has_chartevents_data=1
                ORDER BY subject_id, admittime ASC
                LIMIT 100
            )
        )
        AND subquery1.valuenum IS NOT NULL
    ) AS mainquery
ON mainquery.hadm_id=prescriptions.hadm_id;