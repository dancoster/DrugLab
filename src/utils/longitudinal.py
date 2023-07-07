from src.parsers import mimic, hirid
from src.modeling import discovery, plots, querier
from src.utils import constants

import pandas as pd
import os

def merge_hirid_mimic(mimic_extract, hirid_extract):
    """Combine mimic and hirid longitudinal data

    Args:
        mimci_extract (_type_): _description_
        hirid_extract (_type_): _description_
    """
    hirid_mimic_mapping = constants.HIRID_MIMIC_FEATURE_MAPPING
    
    hirid_m_extract = hirid_extract.rename(columns=hirid_mimic_mapping)
    
    hirid_m_extract = hirid_m_extract.drop(columns=constants.IN_HIRID_NOT_MIMIC_COLS).rename(columns={
        "CHARTTIME":"charttime",
        "EST_DISCHTIME": "DISCHTIME",
    })
    
    mimic_m_extract = mimic_extract.drop(columns=constants.IN_MIMIC_NOT_HIRID_COLS)
    
    final_mimic_long = mimic_m_extract[constants.EXTRACT_COLS]
    final_hirid_long = hirid_m_extract[constants.EXTRACT_COLS]
    
    merged = pd.concat([final_hirid_long, final_mimic_long])
    
    return merged

combine_list = [
    ('Hemoglobin [Mass/volume] in Blood', 'Hemoglobin [Mass/volume] in Arterial blood', 'Hemoglobin [Mass/volume] in blood'),
    ('Lactate [Moles/volume] in Venous blood', 'Lactate [Mass/volume] in Arterial blood', 'Lactate [Mass/volume] in blood'),
    ('Invasive diastolic arterial pressure','Non-invasive diastolic arterial pressure', 'Diastolic arterial pressure'),
    ('Invasive systolic arterial pressure','Non-invasive systolic arterial pressure', 'Systolic arterial pressure')
]

def hirid_longitudinal_generator(raw_path, res_path, gender, age_a, age_b):
    """HIRID Longitudinal dataset generator

    Args:
        raw_path (_type_): path containing HiRiD Dataset, specifically the "observation tables" folder with csv part files
        res_path (_type_): output path for intermediate files and final output files
        gender (_type_): Stratification parameter - patients gender
        age_a (_type_): Stratification parameter - minimum age
        age_b (_type_): Stratification parameter - maximum age

    Returns:
        _type_: HIRID Longitudinal data
    """
    
    # Read raw part files and apply pivot operation to get feature values as columns
    tuples = [(i, i+5) for i in range(0, 246, 5)]
    for lab_parts in tuples:
        print(f"Start {lab_parts}....")
        hirid_mapping = constants.HIRID_MAPPING
        hirid_parser = hirid.HiRiDParser(data=raw_path, res=res_path, gender=gender, age_b=age_b, age_a=age_a, load="MANUAL_MAPPING_HIRID")
        h_med1, h_med2, h_labs = hirid_parser.parse(lab_parts=lab_parts)
        for c_lab in combine_list:
            l = list(h_labs.ITEMID.value_counts().keys())
            if c_lab[0] in l and c_lab[1] in l:
                h_labs["ITEMID"] = h_labs.ITEMID.apply(lambda r: c_lab[2] if r==c_lab[0] else r)
                h_labs["ITEMID"] = h_labs.ITEMID.apply(lambda r: c_lab[2] if r==c_lab[1] else r)
            
        lab_with_index = h_labs[["HADM_ID", "AGE", "GENDER", "CHARTTIME", "VALUENUM", "ITEMID"]].pivot_table("VALUENUM", ["HADM_ID", "AGE", "GENDER",  "CHARTTIME"], "ITEMID")
        lab_with_index = lab_with_index.reset_index()
        
        # Query and add age, geneder, admit and dicharge time of patients.
        # Note: As discharge time data is missing in hirid - estimated discharge time is generated as 15 hours after the last lab test reading.
        disch_time = lab_with_index.groupby(["HADM_ID", "AGE", "GENDER"]).last()
        disch_time["EST_DISCHTIME"] = disch_time["CHARTTIME"]+pd.Timedelta(hours=15)
        lab_with_index = lab_with_index.groupby(["HADM_ID", "AGE", "GENDER", lab_with_index.CHARTTIME.dt.date, lab_with_index.CHARTTIME.dt.hour]).mean()
        
        final_merged = pd.merge(lab_with_index.reset_index(4).rename(columns={"CHARTTIME":"HOUR"}).reset_index().rename(columns={"CHARTTIME":"DATE"}), disch_time.reset_index()[["HADM_ID", "AGE", "GENDER", "EST_DISCHTIME"]], on=["HADM_ID", "AGE", "GENDER"])
        final_merged["HOUR"] = final_merged.HOUR.astype(int).apply(lambda h: f"0{h}" if h<10 else f"{h}")
        final_merged["DATE"] = final_merged.DATE.astype(str)
        final_merged["CHARTTIME"] = pd.to_datetime(final_merged[["DATE", "HOUR"]].agg(' '.join, axis=1), format="%Y-%m-%d %H")
        
        death_data = hirid_parser.g_table.rename(columns={"patientid":"HADM_ID"})[["HADM_ID", "discharge_status"]]
        final_merged = pd.merge(final_merged, death_data)
        final_merged["EST_DISCHTIME"] = pd.to_datetime(final_merged["EST_DISCHTIME"])
        
        # Label is generated for moratality in the first 48 hours after admission 
        def get_label(row, hours=48):
            if row["discharge_status"]=="dead" and row["EST_DISCHTIME"]-row["CHARTTIME"] < pd.Timedelta(hours=hours):
                return 1
            else:
                return 0
        final_merged["LABEL_48"] = final_merged.apply(lambda row: get_label(row, hours=48), axis=1)
        
        final_merged = final_merged.drop(columns=["Amylase [Enzymatic activity/volume] in Body fluid","Creatinine [Moles/volume] in Urine","Glucose [Moles/volume] in Cerebral spinal fluid","Lactate [Moles/volume] in Cerebral spinal fluid", 'Metronidazole tabl 200 mg'])
        final_merged.to_csv(os.path.join(res_path, f"hirid_extract_with_labels_48_new_{lab_parts}.csv"))
        print(f"Done with {lab_parts}.")

    parts = [(i, i+5) for i in range(0, 246, 5)]
    paths = []
    for lab_parts in parts:
        paths.append(os.path.join(res_path, f"hirid_extract_with_labels_48_new_{lab_parts}.csv"))
    final_all_merged = pd.concat([pd.read_csv(path).drop(columns="Unnamed: 0") for path in paths])
    
    final_all_merged.to_csv(os.path.join(res_path, f"hirid_extract_with_labels_48_all_parts_new.csv"))

    return final_merged[['HADM_ID', 'AGE', 'GENDER', 'CHARTTIME', 'discharge_status', 'EST_DISCHTIME', 'LABEL_48',
        'Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma',
        'Albumin [Mass/volume] in Serum or Plasma',
        'Amylase [Enzymatic activity/volume] in Body fluid',
        'Amylase [Enzymatic activity/volume] in Serum or Plasma',
        'Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma',
        'Bicarbonate [Moles/volume] in Arterial blood',
        'Bilirubin.direct [Mass/volume] in Serum or Plasma',
        'Bilirubin.total [Moles/volume] in Serum or Plasma',
        'Calcium [Moles/volume] in Blood',
        'Calcium.ionized [Moles/volume] in Blood',
        'Carboxyhemoglobin/Hemoglobin.total in Arterial blood',
        'Chloride [Moles/volume] in Blood', 'Core body temperature',
        'Creatinine [Moles/volume] in Blood',
        'Creatinine [Moles/volume] in Urine',
        'Glucose [Moles/volume] in Cerebral spinal fluid',
        'Glucose [Moles/volume] in Serum or Plasma', 'Heart rate',
        'Hemoglobin [Mass/volume] in Arterial blood',
        'Hemoglobin [Mass/volume] in Blood',
        'INR in Blood by Coagulation assay',
        'Invasive diastolic arterial pressure',
        'Invasive systolic arterial pressure',
        'Lactate [Mass/volume] in Arterial blood',
        'Lactate [Moles/volume] in Cerebral spinal fluid',
        'Lactate [Moles/volume] in Venous blood',
        'Lymphocytes [#/volume] in Blood', 'Magnesium [Moles/volume] in Blood',
        'Methemoglobin/Hemoglobin.total in Arterial blood',
        'Metronidazole tabl 200 mg', 'Neutrophils/100 leukocytes in Blood',
        'Non-invasive diastolic arterial pressure',
        'Non-invasive systolic arterial pressure',
        'Peripheral oxygen saturation', 'Platelets [#/volume] in Blood',
        'Potassium [Moles/volume] in Blood',
        'Pulmonary artery diastolic pressure',
        'Pulmonary artery systolic pressure', 'Respiratory rate',
        'Sodium [Moles/volume] in Blood']].set_index(['HADM_ID', 'AGE', 'GENDER', 'CHARTTIME'])