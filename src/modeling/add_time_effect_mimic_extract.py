import pandas as pd
import pickle

class AddMedEffect:

    def __init__(self, vitals_data, dataset, label_path=None, admissions=None):
        self.vitals_data = vitals_data
        self.dataset = dataset
        self.admissions = admissions if admissions is not None else self.dataset.admissions
        self.admissions['ADMITTIME'] = pd.to_datetime(self.admissions['ADMITTIME'])

        path = label_path if label_path is not None else '/Users/yanpavan/Desktop/Personal/TAU/DrugLab/data/mimic_extract/itemid_to_variable_map.csv'
        self.meds_label_map = pd.read_csv(path)
        self.meds_label_map = self.meds_label_map[['LEVEL2', 'MIMIC LABEL']]

        self.meds = self._preprocess_meds_data(self.admissions, self.dataset.patient_presc['inputevents'])
        self.vitals_labtests = vitals_data.columns.to_frame()['LEVEL2'].unique()
        self.vals = []

    def editDistance(self, str1, str2):
        len1 = len(str1)
        len2 = len(str2)
    
        DP = [[0 for i in range(len1 + 1)]
                for j in range(2)]
    
        for i in range(0, len1 + 1):
            DP[0][i] = i
    
        for i in range(1, len2 + 1):
            
            for j in range(0, len1 + 1):
                if (j == 0):
                    DP[i % 2][j] = i
    
                elif(str1[j - 1] == str2[i-1]):
                    DP[i % 2][j] = DP[(i - 1) % 2][j - 1]

                else:
                    DP[i % 2][j] = (1 + min(DP[(i - 1) % 2][j],
                                        min(DP[i % 2][j - 1],
                                    DP[(i - 1) % 2][j - 1])))

        return DP[len2 % 2][len1]

    def are_same(self, str1, str2, threshold=0.25):
        str1 = str1.lower()
        str2 = str2.lower()
        m = len(str1)
        n = len(str2)

        return self.editDistance(str1, str2) / min(m,n) >= threshold

    def check_change_labtest_edit_distance(self, row):
        min_edit = 100000
        final_name = row
        for labtest in self.vitals_labtests:
            k = self.editDistance(row, labtest)
            if k<min_edit:
                min_edit = k
                final_name = labtest
        return final_name
    
    def check_change_labtest(self, row):
        if row in self.vitals_labtests:
            return row
        if row.lower() in self.vitals_labtests:
            return row.lower()
        k = self.meds_label_map
        query = k[k['MIMIC LABEL']==row]
        if query.shape[0]>0:
            return query['LEVEL2'].iloc[0]
        else:
            self.check_change_labtest_edit_distance(row)

    def _preprocess_meds_data(self, admissions, meds):
        admit_time = admissions[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME']]
        meds['ENDTIME'] = pd.to_datetime(meds['ENDTIME'])
        temp_data = pd.merge(meds, admit_time, on='HADM_ID', how='inner')
        meds['time_in'] = temp_data['ENDTIME']-temp_data['ADMITTIME']
        meds['hours_in'] = meds['time_in'].apply(lambda x : x.total_seconds()//3600)
        return meds

    def _get_admit_time(self, admit_time, row):
        return admit_time[admit_time['HADM_ID']==row['HADM_ID']]['ADMITTIME'].iloc[0]

    def get_hours(self, time):
        return time.total_seconds()//3600

    def check_subs_with_meds(self, row, meds, sig_meds):
        adm = row.name[1]   # ['hadm_id']
        labtest_hours = row.name[3]     # ['hours_in']

        if adm in meds['HADM_ID'].to_list():
            meds = meds[meds['HADM_ID']==adm]
            # Calculate difference between relative medication time (time from start of admission) and relative time of labtest to get time from prescription
            temp = meds.apply(lambda r : (labtest_hours-r['hours_in'])>0 and (labtest_hours-r['hours_in'])==sig_meds[sig_meds['Medication']==r['LABEL']]['Time'].iloc[0], axis=1) 
            meds = meds[temp]
            return meds.shape[0]

        return 0

    def get_subs_with_meds(self, subjects_vitals, sig_meds, vitals_lab):

        all_meds = self.meds
        meds = all_meds[all_meds['HADM_ID'].isin(subjects_vitals['hadm_id'])]     # filter vital subjects
        meds = meds[meds['LABEL'].isin(sig_meds['Medication'])]    # Is a sig med
        final_subs = vitals_lab.apply(lambda row : self.check_subs_with_meds(row, meds, sig_meds) if row['count']>0 else 0, axis=1)

        return final_subs

    def add_med_effect(self, table):
        vitals_data = self.vitals_data
        labtests_vitals = vitals_data.columns.to_frame()['LEVEL2'].value_counts().keys().to_list()
        table['Lab Tests'] = table['Lab Tests'].apply(self.check_change_labtest)

        res = {}

        # count = 0
        # for labtest in labtests_vitals:
        #     if labtest not in table['Lab Tests'].to_list():
        #         count+=1
        #         continue
        
        for labtest in table['Lab Tests'].to_list():

            print(f'Labtest : {labtest}')

            if labtest in res.keys():
                continue

            sig_labtest_meds = table[table['Lab Tests']==labtest]
            sig_meds = sig_labtest_meds[['Medication', 'Time']]

            data_with_labtest_val = vitals_data[vitals_data[labtest]['count']>0]
            subjects_vitals = data_with_labtest_val.index.to_frame()[['subject_id', 'hadm_id', 'hours_in']]
            

            subs_with_meds = self.get_subs_with_meds(
                subjects_vitals=subjects_vitals,
                vitals_lab=vitals_data[labtest],
                sig_meds=sig_meds            
            )

            print(f'{subs_with_meds[subs_with_meds>0].shape[0]} labtest values updated for {labtest}')
            # labtest_data['sig_counts'] = subs_with_meds

            res[labtest] = {
                'labtest':labtest,
                'counts':subs_with_meds
            }
            pickle.dump(res, open('/Users/yanpavan/Desktop/Personal/TAU/DrugLab/results/mimic_extract/res_label.pkl', 'wb'))

        return res