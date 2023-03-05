from src.parsers import mimic, hirid
from src.modeling import discovery, plots, querier
from src.utils import constants

def setup_io_config(root_path):
    """
    Input - Output config. Add dataset paths
    :root_path -> Repo path which contains 'data' and 'res' folders
    """

    # MIMIC
    data = f"{root_path}/data"
    res = f"{root_path}/results"

    # HIRID
    raw_path = f'{root_path}/data/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/'
    res_path = f'{root_path}/data/hirid-a-high-time-resolution-icu-dataset-1.1.1'
    
    return data, res, raw_path, res_path

def setup_stratification_config():
    gender="MF"
    age_b=40
    age_a=80 
    ethnicity="WHITE" 
    lab_mapping=constants.LAB_MAPPING
    before_windows = [(0,12), (0,6)]
    after_windows = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12)]
    return gender, age_a, age_b, ethnicity, lab_mapping, before_windows, after_windows

def mimic_analysis(data, res, gender, age_a, age_b, ethnicity, lab_mapping, before_windows, after_windows):
    # MIMIC
    mimic_parser = mimic.MIMICParser(data=data, res=res, gender=gender, age_b=age_b, age_a=age_a, ethnicity=ethnicity, lab_mapping=lab_mapping)
    m_med1, m_med2, m_labs = mimic_parser.parse()
    ## Querier
    mimic_data_querier = querier.DatasetQuerier(
        data = data,
        res = res,
        gender=gender, 
        age_b=age_b, 
        age_a=age_a, 
        ethnicity=ethnicity, 
        lab_mapping=lab_mapping
    )
    m_final_lab_med_data = mimic_data_querier.generate_med_lab_data(m_labs, m_med1, m_med2, before_windows, after_windows)
    ## Discovery
    discovery.ClinicalDiscoveryAnalysis()
    ## Plots
    plotter = plots.ClinicalPlotAnalysis(
        data = data,
        res = res,
        gender=gender, 
        age_b=age_b, 
        age_a=age_a, 
        ethnicity=ethnicity, 
        lab_mapping=lab_mapping
    )
    m_corrs_data_df = plotter.plot(m_final_lab_med_data, m_labs, before_windows=before_windows, after_windows=after_windows)
    return mimic_parser, m_med1, m_med2, m_labs, m_final_lab_med_data, plotter, m_corrs_data_df


def hirid_analysis(raw_path, res_path, gender, age_a, age_b, ethnicity, lab_mapping, before_windows, after_windows):
    pass

if __name__=="__main__":
    # IO Config
    root_path = "/Users/pavan/Library/CloudStorage/GoogleDrive-f20190038@hyderabad.bits-pilani.ac.in/My Drive/TAU/Code/DrugLab"
    data, res, raw_path, res_path = setup_io_config(root_path=root_path)
    
    # Stratification Config
    gender, age_a, age_b, ethnicity, lab_mapping, before_windows, after_windows = setup_stratification_config()

    # MIMIC
    mimic_parser, m_med1, m_med2, m_labs, m_final_lab_med_data, plotter, m_corrs_data_df = mimic_analysis(data, res, raw_path, res_path, gender, age_a, age_b, ethnicity, lab_mapping, before_windows, after_windows)
    
    # HIRID
    hirid_mapping = constants.HIRID_MAPPING
    hirid_parser = hirid.HiRiDParser(data=raw_path, res=res_path, gender=gender, age_b=age_b, age_a=age_a)
    h_med1, h_med2, h_labs = hirid_parser.parse()
    lab_ids = [l for k in hirid_mapping.values() for l in k]
    h_labs_1 = h_labs[h_labs.OldITEMID.isin(lab_ids)]
    
    hirid_data_querier = querier.DatasetQuerier(
        data = raw_path,
        res = res_path,
        gender=gender, 
        age_b=age_b, 
        age_a=age_a, 
        ethnicity=ethnicity, 
    )
    final_h_final_lab_med_data, raw_h_final_lab_med_data = hirid_data_querier.generate_med_lab_data(h_labs_1, h_med1, h_med2, before_windows, after_windows)
    
    h_plotter = plots.ClinicalPlotAnalysis(
        data = raw_path,
        res = res_path,
        gender=gender, 
        age_b=age_b, 
        age_a=age_a, 
        ethnicity="", 
        lab_mapping={}
    )
    h_corrs_data_df = h_plotter.plot(final_h_final_lab_med_data, h_labs, before_windows=before_windows, after_windows=after_windows)
    
    