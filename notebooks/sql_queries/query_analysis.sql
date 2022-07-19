SELECT * FROM (
    (
        SELECT hadm_id FROM (
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
            )
        )
        AND subquery1.valuenum IS NOT NULL
    ) as q1
    JOIN
    (
        SELECT * FROM (
            SELECT * FROM labevents INNER JOIN (SELECT * FROM d_labitems) k
            ON labevents.itemid=k.itemid
        ) AS subquery1
        WHERE
        subquery1.hadm_id IS NOT NULL
        AND
        (
            subquery1.hadm_id IN (
                SELECT DISTINCT ON (subject_id) hadm_id FROM admissions
                WHERE has_chartevents_data=1
                ORDER BY subject_id, admittime ASC
            )
        )
        AND subquery1.valuenum IS NOT NULL
    ) as q2
    on q1.hadm_id=q2.hadm_id
) AS query1
WHERE DATEDIFF('day', query1.startdate, query1.charttime) BETWEEN 0 AND 1
    AND 
    DATEDIFF('day', query1.charttime, query1.enddate) BETWEEN 0 AND 1;