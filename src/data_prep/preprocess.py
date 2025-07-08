import pandas as pd
from datetime import datetime
import os

DIR = 'src/'
ACTUAL_DIR = 'src/test_T1_actual'
TRAIN_T1_DIR = 'src/train_T1'

mock_cust_df = pd.read_csv('src/mock_customers.csv')
test_df = pd.read_csv('src/clustering/data_v3/test_df.csv')
train_df = pd.read_csv('src/clustering/data_v3/train_df.csv')


col_selected_T0 = ['DOB_prev', 'NO_OF_CHLD_prev', 'NO_OF_VHCL_prev',
       'GNDR_CD_prev', 'EDU_DESC_prev', 'MRRY_DESC_prev', 'OCPN_DESC_prev',
       'ADDR_PRVC_prev', 'ADDR_ZIP_prev', 'SCB_PROD_HLDNG_CNT.1',
       'SCB_PROD_SUB_GRP_HLDNG_CNT.1', 'PROD_GRP_WLTH_HLDNG_CNT.1',
       'PROD_GRP_LENDING_HLDNG_CNT.1', 'PROD_GRP_PYMT_HLDNG_CNT.1',
       'PROD_GRP_SRVC_HLDNG_CNT.1', 'PROD_GRP_BSNS_LENDING_HLDNG_CNT.1',
       'DPST_ACCT_CNT_prev', 'DPST_END_BAL_prev', 'DPST_TXN_CNT_prev',
       'DPST_TXN_CNT_AVG_prev', 'DPST_TXN_CNT_MIN_prev',
       'DPST_TXN_CNT_MAX_prev', 'DPST_INFLW_CNT_prev',
       'DPST_INFLW_CNT_MIN_prev', 'DPST_INFLW_CNT_MAX_prev',
       'DPST_OTFLW_CNT_prev', 'DPST_OTFLW_CNT_MIN_prev',
       'DPST_OTFLW_CNT_MAX_prev', 'DPST_INFLW_AMT_TOT_prev',
       'DPST_OTFLW_AMT_TOT_prev']

col_map_T0 = {
    'DOB_prev': 'Date of birth',
    'NO_OF_CHLD_prev': 'Number of Children',
    'NO_OF_VHCL_prev': 'Number of Vehicles',
    'GNDR_CD_prev': 'Gender',
    'EDU_DESC_prev': 'Education level',
    'MRRY_DESC_prev': 'Marital status',
    'OCPN_DESC_prev': 'Occupation',
    'ADDR_PRVC_prev': 'Province',
    'ADDR_ZIP_prev': 'Postal Code',
    'SCB_PROD_HLDNG_CNT.1': 'Savings Account',
    'SCB_PROD_SUB_GRP_HLDNG_CNT.1': 'Savings Account Subgroup',
    'PROD_GRP_WLTH_HLDNG_CNT.1': 'Health Insurance',
    'PROD_GRP_LENDING_HLDNG_CNT.1': 'Lending',
    'PROD_GRP_PYMT_HLDNG_CNT.1': 'Payment',
    'PROD_GRP_SRVC_HLDNG_CNT.1': 'Service',
    'PROD_GRP_BSNS_LENDING_HLDNG_CNT.1': 'Business Lending',
    'DPST_ACCT_CNT_prev': 'Deposit Account',
    'DPST_END_BAL_prev': 'Deposit Account Balance',
    'DPST_TXN_CNT_prev': 'Deposit Account Transactions',
    'DPST_TXN_CNT_AVG_prev': 'Deposit Account Transactions AVG',
    'DPST_TXN_CNT_MIN_prev': 'Deposit Account Transactions MIN',
    'DPST_TXN_CNT_MAX_prev': 'Deposit Account Transactions MAX',
    'DPST_INFLW_CNT_prev': 'Deposit Account Inflow',
    'DPST_INFLW_CNT_MIN_prev': 'Deposit Account Inflow MIN',
    'DPST_INFLW_CNT_MAX_prev': 'Deposit Account Inflow MAX',
    'DPST_OTFLW_CNT_prev': 'Deposit Account Outflow',
    'DPST_OTFLW_CNT_MIN_prev': 'Deposit Account Outflow MIN',
    'DPST_OTFLW_CNT_MAX_prev': 'Deposit Account Outflow MAX',
    'DPST_INFLW_AMT_TOT_prev': 'Deposit Account Inflow Amount',
    'DPST_OTFLW_AMT_TOT_prev': 'Deposit Account Outflow Amount'
}

col_selected_T1 = ['DOB', 'NO_OF_CHLD', 'NO_OF_VHCL',
       'GNDR_CD', 'EDU_DESC', 'MRRY_DESC', 'OCPN_DESC',
       'ADDR_PRVC', 'ADDR_ZIP', 'SCB_PROD_HLDNG_CNT',
       'SCB_PROD_SUB_GRP_HLDNG_CNT', 'PROD_GRP_WLTH_HLDNG_CNT',
       'PROD_GRP_LENDING_HLDNG_CNT', 'PROD_GRP_PYMT_HLDNG_CNT',
       'PROD_GRP_SRVC_HLDNG_CNT', 'PROD_GRP_BSNS_LENDING_HLDNG_CNT',
       'DPST_ACCT_CNT', 'DPST_END_BAL', 'DPST_TXN_CNT',
       'DPST_TXN_CNT_AVG', 'DPST_TXN_CNT_MIN',
       'DPST_TXN_CNT_MAX', 'DPST_INFLW_CNT',
       'DPST_INFLW_CNT_MIN', 'DPST_INFLW_CNT_MAX',
       'DPST_OTFLW_CNT', 'DPST_OTFLW_CNT_MIN',
       'DPST_OTFLW_CNT_MAX', 'DPST_INFLW_AMT_TOT',
       'DPST_OTFLW_AMT_TOT']

col_map_T1 = {
    'DOB': 'Date of birth',
    'NO_OF_CHLD': 'Number of Children',
    'NO_OF_VHCL': 'Number of Vehicles',
    'GNDR_CD': 'Gender',
    'EDU_DESC': 'Education level',
    'MRRY_DESC': 'Marital status',
    'OCPN_DESC': 'Occupation',
    'ADDR_PRVC': 'Province',
    'ADDR_ZIP': 'Postal Code',
    'SCB_PROD_HLDNG_CNT': 'Savings Account',
    'SCB_PROD_SUB_GRP_HLDNG_CNT': 'Savings Account Subgroup',
    'PROD_GRP_WLTH_HLDNG_CNT': 'Health Insurance',
    'PROD_GRP_LENDING_HLDNG_CNT': 'Lending',
    'PROD_GRP_PYMT_HLDNG_CNT': 'Payment',
    'PROD_GRP_SRVC_HLDNG_CNT': 'Service',
    'PROD_GRP_BSNS_LENDING_HLDNG_CNT': 'Business Lending',
    'DPST_ACCT_CNT': 'Deposit Account',
    'DPST_END_BAL': 'Deposit Account Balance',
    'DPST_TXN_CNT': 'Deposit Account Transactions',
    'DPST_TXN_CNT_AVG': 'Deposit Account Transactions AVG',
    'DPST_TXN_CNT_MIN': 'Deposit Account Transactions MIN',
    'DPST_TXN_CNT_MAX': 'Deposit Account Transactions MAX',
    'DPST_INFLW_CNT': 'Deposit Account Inflow',
    'DPST_INFLW_CNT_MIN': 'Deposit Account Inflow MIN',
    'DPST_INFLW_CNT_MAX': 'Deposit Account Inflow MAX',
    'DPST_OTFLW_CNT': 'Deposit Account Outflow',
    'DPST_OTFLW_CNT_MIN': 'Deposit Account Outflow MIN',
    'DPST_OTFLW_CNT_MAX': 'Deposit Account Outflow MAX',
    'DPST_INFLW_AMT_TOT': 'Deposit Account Inflow Amount',
    'DPST_OTFLW_AMT_TOT': 'Deposit Account Outflow Amount'
}

def preprocess(mock_cust_df, col_selected, col_map, file_name, dir):
    df = mock_cust_df.copy()
    cust_ids = df['cust_id']
    # Change col names
    df = df[col_selected]


    df = df.rename(columns=col_map)

    # Descriptive 
    df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})

    # Clean
    df['Province'] = df['Province'].replace({'msn': 'แม่ฮ่องสอน', 'pkn': 'ประจวบคีรีขันธ์', 'nsn': 'นครสวรรค์', 'brm': 'บุรีรัมย์'})

    # Fill nan
    df['Savings Account'] = df['Savings Account'].fillna(0.0)
    df['Savings Account Subgroup'] = df['Savings Account Subgroup'].fillna(0.0)
    df['Health Insurance'] = df['Health Insurance'].fillna(0.0)
    df['Lending'] = df['Lending'].fillna(0.0)
    df['Payment'] = df['Payment'].fillna(0.0)
    df['Service'] = df['Service'].fillna(0.0)
    df['Business Lending'] = df['Business Lending'].fillna(0.0)
    df['Deposit Account'] = df['Deposit Account'].fillna(0.0)

    # Add Age
    df['Date of birth'] = pd.to_datetime(df['Date of birth'], errors='coerce')
    current_date = datetime(2023, 1, 31)
    df['Age'] = (current_date - df['Date of birth']).dt.days // 365

    # Add Region
    province_to_region = {
        'ปราจีนบุรี': 'Central',
        'นครราชสีมา': 'Northeastern',
        'ปทุมธานี': 'Central',
        'แม่ฮ่องสอน': 'Northern',
        'สมุทรสงคราม': 'Central',
        'พระนครศรีอยุธยา': 'Central',
        'กรุงเทพมหานคร': 'Central',
        'ลำปาง': 'Northern',
        'เชียงราย': 'Northern',
        'สุรินทร์': 'Northeastern',
        'อุบลราชธานี': 'Northeastern',
        'นครปฐม': 'Central',
        'สงขลา': 'Southern',
        'นนทบุรี': 'Central',
        'ชัยภูมิ': 'Northeastern',
        'ชลบุรี': 'Eastern',
        'นครศรีธรรมราช': 'Southern',
        'กาญจนบุรี': 'Western',
        'สุพรรณบุรี': 'Central',
        'เชียงใหม่': 'Northern',
        'สมุทรปราการ': 'Central',
        'ชุมพร': 'Southern',
        'ฉะเชิงเทรา': 'Eastern',
        'มหาสารคาม': 'Northeastern',
        'พะเยา': 'Northern',
        'กระบี่': 'Southern',
        'ระยอง': 'Eastern',
        'เพชรบุรี': 'Western',
        'อุทัยธานี': 'Central',
        'สมุทรสาคร': 'Central',
        'สุราษฎร์ธานี': 'Southern',
        'ตราด': 'Eastern',
        'ตรัง': 'Southern',
        'เพชรบูรณ์': 'Central',
        'ภูเก็ต': 'Southern',
        'พัทลุง': 'Southern',
        'ร้อยเอ็ด': 'Northeastern',
        'ปัตตานี': 'Southern',
        'หนองคาย': 'Northeastern',
        'ขอนแก่น': 'Northeastern',
        'บุรีรัมย์': 'Northeastern',
        'นราธิวาส': 'Southern',
        'เลย': 'Northeastern',
        'น่าน': 'Northern',
        'ศรีสะเกษ': 'Northeastern',
        'หนองบัวลำภู': 'Northeastern',
        'จันทบุรี': 'Eastern',
        'ประจวบคีรีขันธ์': 'Western',
        'สตูล': 'Southern',
        'นครสวรรค์': 'Central',
        'แพร่': 'Northern',
        'ยะลา': 'Southern',
        'กำแพงเพชร': 'Central',
        'พิษณุโลก': 'Northern',
        'ลำพูน': 'Northern',
        'ระนอง': 'Southern',
        'นครพนม': 'Northeastern',
        'สระแก้ว': 'Eastern',
        'ยโสธร': 'Northeastern',
        'ลพบุรี': 'Central',
        'อุดรธานี': 'Northeastern',
        'ราชบุรี': 'Western',
        'สิงห์บุรี': 'Central',
        'กาฬสินธุ์': 'Northeastern',
        'พังงา': 'Southern',
        'สระบุรี': 'Central',
        'สกลนคร': 'Northeastern',
        'บึงกาฬ': 'Northeastern',
        'อำนาจเจริญ': 'Northeastern',
        'พิจิตร': 'Central',
        'ตาก': 'Northern',
        'อุตรดิตถ์': 'Northern',
        'ชัยนาท': 'Central',
        'นครนายก': 'Central',
        'สุโขทัย': 'Northern',
        'มุกดาหาร': 'Northeastern',
        'อ่างทอง': 'Central',
        # Handle missing or NaN case
        float('nan'): 'Unknown'
    }


    df['Region'] = df['Province'].apply(lambda x: province_to_region.get(x, 'Unknown'))

    df.fillna('Unknown', inplace=True)

    # Add occupation group
    occupation_groups = {
        # Corporate Employees
        "industrial business employee": "Corporate Employee",
        "commerce employee": "Corporate Employee",
        "service business employee": "Corporate Employee",
        "employee of company, store, office": "Corporate Employee",
        "financial institution employee": "Corporate Employee",
        "commercial bank employee": "Corporate Employee",
        "agro industry employee": "Corporate Employee",
        "state enterprise employee": "Corporate Employee",
        "government employee": "Corporate Employee",
        "civil servant, state enterprise employee, pensioner": "Corporate Employee",
        "employee of foundation": "Corporate Employee",
        "employee of association": "Corporate Employee",
        "employee of cooperative": "Corporate Employee",
        "employee/ group in non-profit organization": "Corporate Employee",

        # Entrepreneurs & Business Owners
        "commerce entrepreneur": "Entrepreneur",
        "service business entrepreneur": "Entrepreneur",
        "industrial entrepreneur": "Entrepreneur",
        "agricultural/ agro industry entrepreneur": "Entrepreneur",
        "entrepreneur/ owner/ company director/ partner": "Entrepreneur",
        "businessman": "Entrepreneur",
        "traders in gems / precious metals, goldsmiths": "Entrepreneur",

        # Professionals
        "engineer": "Professional",
        "architect": "Professional",
        "doctor": "Professional",
        "dentist": "Professional",
        "pharmacist": "Professional",
        "nurse": "Professional",
        "medical technologist": "Professional",
        "accountant (with certificate, diploma, university degree)": "Professional",
        "lawyer and legal practitioner": "Professional",
        "education practitioner": "Professional",
        "teacher and executive in university or equivalent institution": "Professional",
        "teacher and executive in kindergarten; primary, secondary, vocational school": "Professional",
        "judicial officer (judge)": "Professional",
        "public attorney": "Professional",
        "politician": "Professional",

        # Creative & Specialized Freelancers
        "freelance": "Freelancer",
        "other freelancers": "Freelancer",
        "designer, painter, photographer and other artists": "Freelancer",
        "mass communication and writing professional": "Freelancer",

        # Students
        "student, university student": "Student",

        # Homemakers & Unemployed
        "housewife/househusband": "Homemaker",
        "unemployed": "Unemployed",

        # Agriculture & Trade
        "merchant, street vendor": "Agriculture/Trade",
        "trade of antiques under control of sale by auction and trade of antiques act b.e.2474 (1931)": "Agriculture/Trade",

        # Others / Rare
        "travelling agency both domestic and international": "Other",
        "professionals otherwise specified or unknown occupation": "Other",
        "soldier": "Other",
        "police officer": "Other"
    }
    df['Occupation Group'] = df['Occupation'].map(occupation_groups).fillna('Other')

    # Drop DoB, Province, Postcode, Occupation
    df = df.drop(['Date of birth', 'Province', 'Postal Code', 'Occupation'], axis=1)

    # Save csv
    result_filename = file_name
    result_path = os.path.join(dir, result_filename)

    df['CUST_ID'] = cust_ids
    df.to_csv(result_path, index=False)

    print(f"\n✅ Results saved to: {result_path}")


# clean mock
# preprocess(mock_cust_df, col_selected_T0, col_map_T0, "cleaned_mock.csv", DIR)

# test T1 actual
# filtered_mock = mock_cust_df[mock_cust_df['cust_id'].isin(test_df['CUST_ID'])]
# preprocess(filtered_mock, col_selected_T1, col_map_T1, "test_T1_actual_v3.csv", ACTUAL_DIR)

# train T1
# filtered_mock = mock_cust_df[mock_cust_df['cust_id'].isin(train_df['CUST_ID'])]
# preprocess(filtered_mock, col_selected_T1, col_map_T1, "train_T1_v3.csv", TRAIN_T1_DIR)