# -*- coding: utf-8 -*-
import requests
import zipfile
import logging
import re
import csv
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import time
from lxml import html
from bs4 import BeautifulSoup
from functools import partial

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from itertools import product

from multiprocessing.pool import ThreadPool

def download_lending_club():
    """ Downloads `Lending Club` data from the available sources
        - https://resources.lendingclub.com/LoanStats3a.csv.zip
        - https://resources.lendingclub.com/LoanStats3b.csv.zip
        - https://resources.lendingclub.com/LoanStats3c.csv.zip
        - https://resources.lendingclub.com/LoanStats3d.csv.zip
        - https://resources.lendingclub.com/LCDataDictionary.xlsx
    """
    # https://www.sec.gov/edgar/search/?r=el#/q=%2522(Sales%2520Report)%2522&dateRange=custom&category=form-cat0&ciks=0001409970&entityName=LendingClub%2520Corp%2520(CIK%25200001409970)&startdt=2016-01-16&enddt=2016-10-16&sort=desc
    
    logger = logging.getLogger(__name__)
    lst_files = ['LoanStats3a.csv.zip', 'LoanStats3b.csv.zip', 'LoanStats3c.csv.zip', 'LoanStats3d.csv.zip']
    lst_files += [f'LoanStats_{year}Q{quarter}.csv.zip' for year, quarter in product(range(2016, 2025), range(1, 5))]
    lst_files += ['LCDataDictionary.xlsx']

    url_prefix = 'https://resources.lendingclub.com'
    path_out = Path('data/external/lending-club')
    path_out.mkdir(parents=True, exist_ok=True)
    for file in lst_files:
        path_zip = path_out / file
        if path_zip.is_file():
            logger.info(f'file {file} already exists')
            continue

        logger.info('downloading {file}')
        url_data = url_prefix + '/' + file
        response = requests.get(url_data)
        response.raise_for_status()
        with open(path_zip, 'wb') as f:
            f.write(response.content)
            
    print('Download Lending Club finished')

def extract_lending_club():
    """ Extracts lending club data
    """
    logger = logging.getLogger(__name__)
    # url_data = 'https://archive.ics.uci.edu/static/public/28/japanese+credit+screening.zip'
    # Send a GET request to the URL
    path_out = Path('data/external/lending-club')
    path_out.mkdir(parents=True, exist_ok=True)
    for path_zip in path_out.glob('*.zip'):
        with zipfile.ZipFile(path_zip, 'r') as zip_ref:
            zip_ref.extractall(path_out)

def concatenate_lending_club():
    path_inp = Path('data/external/lending-club/')
    lst_files = ['LoanStats3a.csv.zip', 'LoanStats3b.csv.zip', 'LoanStats3c.csv.zip', 'LoanStats3d.csv.zip']
    lst_files += [f'LoanStats_{year}Q{quarter}.csv.zip' for year, quarter in product(range(2016, 2025), range(1, 5))]
 
    df_common = pd.DataFrame()  # Initialize an empty DataFrame
    for file in lst_files:
        print(file)
        try:
            zip_file = path_inp / file
            df = pd.read_csv(zip_file, compression='zip', header=0, sep=',', 
                            quotechar='"', skip_blank_lines=True, 
                            skipfooter=2, skiprows=1, engine='python')
            df_common = pd.concat([df_common, df], ignore_index=True)
        except Exception as e:
            print(f"Error processing file '{file}': {e}")
    df_common.to_csv(path_inp / 'LoanStats.csv', index=False)


dic_types = {
'id':str,
'member_id':str,
'loan_amnt':float,
'funded_amnt':float,
'funded_amnt_inv':float,
'term':'category',
'int_rate': str,
'installment':float,
'grade':'category',
'sub_grade':'category',
'emp_title':str,
'emp_length':'category',
'home_ownership':'category',
'annual_inc':float,
'verification_status':'category',
'issue_d': str,
'loan_status':'category',
'pymnt_plan':'category',
'url':str,
'desc':str,
'purpose':'category',
'title': str,
'zip_code': 'category',
'addr_state' : 'category',
'dti' :float,
'delinq_2yrs' : float,
'earliest_cr_line':str,
'inq_last_6mths' : float,
'mths_since_last_delinq':float,
'mths_since_last_record':float,
'open_acc':float,
'pub_rec':float,
'revol_bal':float,
'revol_util':str,
'total_acc':float,
'initial_list_status':'category',
'out_prncp':float,
'out_prncp_inv' : float,
'total_pymnt':float,
'total_pymnt_inv':float,
'total_rec_prncp':float,
'total_rec_int':float,
'total_rec_late_fee':float,
'recoveries':float,
'collection_recovery_fee':float,
'last_pymnt_d':str,
'last_pymnt_amnt':float,
'next_pymnt_d':str,
'last_credit_pull_d':str,
'collections_12_mths_ex_med':float,
'mths_since_last_major_derog': float,
'policy_code': 'category',
'application_type':'category',
'annual_inc_joint':float,
'dti_joint':float,
'verification_status_joint':'category',
'acc_now_delinq':float,
'tot_coll_amt':float,
'tot_cur_bal':float,
'open_acc_6m':float,
'open_act_il':float,
'open_il_12m':float,
'open_il_24m':float,
'mths_since_rcnt_il':float,
'total_bal_il':float,
'il_util':float,
'open_rv_12m':float,
'open_rv_24m':float,
'max_bal_bc':float,
'all_util':float,
'total_rev_hi_lim':float,
'inq_fi':float,
'total_cu_tl':float,
'inq_last_12m':float,
'acc_open_past_24mths':float,
'avg_cur_bal':float,
'bc_open_to_buy':float,
'bc_util':float,
'chargeoff_within_12_mths':float,
'delinq_amnt':float,
'mo_sin_old_il_acct':float,
'mo_sin_old_rev_tl_op':float,
'mo_sin_rcnt_tl':float,
'mort_acc':float,
'mths_since_recent_bc':float,
'mths_since_recent_bc_dlq':float,
'mths_since_recent_inq':float,
'mths_since_recent_revol_delinq':float,
'num_accts_ever_120_pd':float,
'num_actv_bc_tl':float,
'num_actv_rev_tl':float,
'num_bc_sats':float,
'num_bc_tl':float,
'num_il_tl':float,
'num_op_rev_tl':float,
'num_rev_accts':float,
'num_rev_tl_bal_gt_0' :float,
'num_sats':float,
'num_tl_120dpd_2m':float,
'num_tl_30dpd':float,
'num_tl_90g_dpd_24m':float,
'num_tl_op_past_12m':float,
'pct_tl_nvr_dlq':float,
'percent_bc_gt_75':float,
'pub_rec_bankruptcies':float,
'tax_liens':float,
'tot_hi_cred_lim':float,
'total_bal_ex_mort':float,
'total_bc_limit':float,
'total_il_high_credit_limit':float,
'revol_bal_joint':float,
'sec_app_earliest_cr_line': str,
'sec_app_inq_last_6mths':float,
'sec_app_mort_acc':float,
'sec_app_open_acc':float,
'sec_app_revol_util':float,
'sec_app_open_act_il':float,
'sec_app_num_rev_accts':float,
'sec_app_chargeoff_within_12_mths':float,
'sec_app_collections_12_mths_ex_med':float,
'sec_app_mths_since_last_major_derog':float,
'hardship_flag':'category',
'hardship_type':'category',
'hardship_reason':str,
'hardship_status':'category',
'deferral_term':float,
'hardship_amount':float,
'hardship_start_date':str,
'hardship_end_date':str,
'payment_plan_start_date':str,
'hardship_length':float,
'hardship_dpd':float,
'hardship_loan_status':'category',
'orig_projected_additional_accrued_interest':float,
'hardship_payoff_balance_amount':float,
'hardship_last_payment_amount':float,
'debt_settlement_flag':'category',
'debt_settlement_flag_date':str,
'settlement_status':'category',
'settlement_date':str,
'settlement_amount':float,
'settlement_percentage':float,
'settlement_term':float,
}

def clean_percentage_string(value):
    if isinstance(value, str):
        stripped_val = value.replace('%', '').strip()
        try:
            return float(stripped_val)
        except:
            return None
    return value  # Return the value as is if it's not a string

def parse_date(date_str):
    return pd.to_datetime(date_str, format='%b-%Y') #'%b' is the abbreviated month name and '%Y' is the four digit year.

dic_converters = {
    'int_rate': clean_percentage_string,
    'revol_util' : clean_percentage_string,

}
def read_lending_club():
    path_inp = Path('data/external/lending-club/')
    csv_file = path_inp / 'LoanStats.csv'
    lst_cols = list(dic_types.keys())
    lst_cols = [x for x in lst_cols if x not in ['id', 'member_id', 'url', 'pymnt_plan']]
    df = pd.read_csv(csv_file,  usecols = lst_cols, engine='python')
    # for col in ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d'
    #             'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
    #             'debt_settlement_flag_date', 'settlement_date']:
    #     if col not in lst_cols:
    #         continue
    #     df[col] = pd.to_datetime(df[col], format='%b-%Y')
    # for col in ['delinq_2yrs', 'inq_last_6mths']:
    #     if col not in lst_cols:
    #         continue
    #     df[col] = df[col].astype('Int64')
    
    print(df.shape)
    
    return df


def download_sales_lending_club():
    # https://resources.lendingclub.com/reports/secdocs/supplement/sales/salessup_20150908.htm.zip
  
    path_out = Path('data/external/lending-club/sales-reports-zip')
    path_out.mkdir(parents=True, exist_ok=True)
    
    date_format="%Y%m%d"
    start_date_str = '20081022'
    start_date = datetime.datetime.strptime(start_date_str, date_format).date()
    end_date_str = '20210115'
    end_date = datetime.datetime.strptime(end_date_str, date_format).date()
    date = start_date
    date_ok = None
    while date <= end_date:
        date_str = date.strftime(date_format)
        path_zip = path_out / f'salessup_{date_str}.htm.zip'
        url_data = f'https://resources.lendingclub.com/reports/secdocs/supplement/sales/salessup_{date_str}.htm.zip'
        try:
            print('Tutaj', date)
            response = requests.get(url_data)
            response.raise_for_status()
            with open(path_zip, 'wb') as f:
                f.write(response.content)
            print(path_zip)
            date_ok =  date
            date += datetime.timedelta(days=7)
            time.sleep(0.7)
        except:
            print('Wolniej....', date_ok)
            if date_ok is None:
                date = date + datetime.timedelta(days=1)
            else:
                date = date_ok + datetime.timedelta(days=1)
                date_ok = None
            time.sleep(0.7)

def extract_sales_repors_zip():
    zip_path = 'data/external/lending-club/sales-reports-zip'
    out_path = 'data/external/lending-club/sales-reports-htm'
    for zip_file in Path(zip_path).glob('*.zip'):
        print(zip_file)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(out_path)
                        
def read_sales_report_zip():
    zip_filepath = 'data/external/lending-club/sales-reports-zip/salessup_20150908.htm.zip'
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        html_files = [f for f in zip_ref.namelist() if f.endswith(('.html', '.htm'))]
        html_filename = html_files[0]  # Get the name of the only HTML file
        with zip_ref.open(html_filename) as f:
            soup = BeautifulSoup(f, 'html.parser')
            table = soup.find('table').find('tr').find('td').find('p')
            table.text

def loop_sales_reports_html():
    def number_from_sales_report_html(html_filepath):
        # html_filepath = Path('data/external/lending-club/sales-reports-htm') / 'salessup_20190305.htm'
        with open(html_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines(1000)  # Read up to 100 lines
            content = "".join(lines)
            tree = html.fromstring(content)
            title = tree.xpath('//title[1]/text()')[0]
            match = re.search(r"No\. (\d+)", title)  # \d+ matches one or more digits
            prospectus_num  = int(match.group(1))
            
            table = tree.xpath('//table[1]/tr[1]/td[1]')
            prospectus_date_content = table[0].text_content()
            match = re.search(r"Prospectus dated (.*)", prospectus_date_content)  # Capture the date part
            date_string = match.group(1)  # Extract the captured date string
            date_object = datetime.datetime.strptime(date_string, "%B %d, %Y")
            return  (date_object, prospectus_num)
        
    path_inp = Path('data/external/lending-club/sales-reports-htm')
    lst_file_number = [(file.name, *number_from_sales_report_html(file)) for file in path_inp.glob('*.htm')]
    lst_file_number_sorted = sorted(lst_file_number, key=lambda x: (x[1], x[2]))
    lst_file_number_sorted = sorted(lst_file_number, key=lambda x: x[1])
    df = pd.DataFrame(lst_file_number_sorted, columns=['file', 'date', 'prospectus_num'])
    path_out = Path('data/processed/lending-club')
    path_out.mkdir(parents=True, exist_ok=True)
    df.to_excel(path_out/ 'sales_reports.xlsx', index=False)
    

def key_val_to_dict(table_key:list, table_val:list) -> dict:
    """
    Converts two lists into a dictionary. If for a given pair key and value are empty,
    they are skipped.
    """
    dic_out = dict()
    for k, v in zip(table_key, table_val):
        if k =='' and v =='':
            continue
        dic_out[k.rstrip(':')] = v
    return dic_out

def clear_monthly_salary(val:str):
    """
    Clears the format read from the html document by removing denominator
    and dollar currency symbol.
    """
    if isinstance(val, str): # Check if it is a string
        return val.rstrip(' / month').lstrip('$').replace(',', '')
    return val # Return other types of values




def sales_report_from_html_to_excel(html_filepath = None):
    if html_filepath is None:
        html_filepath = Path('data/external/lending-club/sales-reports-htm') / 'salessup_20190305.htm'
    
    excel_filepath = Path('data/external/lending-club/sales-reports-xlsx') / f'{html_filepath.stem}.xlsx'
    if excel_filepath.is_file():
        return
    
    etree = html.parse(html_filepath)
    root = etree.getroot()
    main_div = root.find('body').find('div')
    
    table_prospectus_desc = main_div.find('table')
    text_prospectus_desc = table_prospectus_desc.find('tr').find('td').text_content().strip().replace('\r\n',' ').replace('\n',' ')
   
    re_text_prospectus_desc = re.compile(r'No. (\d+) dated (.+\d{4}) to Prospectus dated (.+\d{4})$')
    suplement_no, suplement_date, prospectus_date =  re_text_prospectus_desc.findall(text_prospectus_desc)[0]
    
    dic_prospectus = {'prospectus_date': pd.to_datetime(prospectus_date, format='%B %d, %Y').date(),
                      'suplement_no': int(suplement_no), 
                      'suplement_date': pd.to_datetime(suplement_date, format='%B %d, %Y').date(),}
                       

    lst_positions = [n for n, x in enumerate(main_div) if str(x.text_content().strip()).startswith('Member Payment Dependent Notes Series')]
    lst_positions += [len(main_div)]
   
    LEN_WAS_REQUESTED = len(' was requested on')
    
    def get_values(iter_elements):
        # find 1st table
        for element in iter_elements:
            if element.tag == 'table':
                table_key = [x.text_content().strip() for x in element.xpath('.//tr[1]/td')]
                table_val = [x.text_content().strip() for x in element.xpath('.//tr[2]/td')]
                break

        for element in iter_elements:
            text_content = element.text_content().strip()
            if element.tag == 'p' and text_content.startswith('This series of Notes was issued upon'):
                pos_was_requested = text_content.find(' was requested on')
                pos_by = text_content.find(' by', pos_was_requested+LEN_WAS_REQUESTED)
                text_date_requested = text_content[pos_was_requested+LEN_WAS_REQUESTED+1:pos_by]
                table_key += ['date_requested']
                table_val += [text_date_requested]
                break
            
        # find 2nd table
        for element in iter_elements:
            if element.tag == 'table':
                table_key += [x.text_content().strip() for x in element.xpath('.//td[position() mod 2 = 1]')]
                table_val += [x.text_content().strip() for x in element.xpath('.//td[position() mod 2 = 0]')]
                break
        
        for element in iter_elements:
            text_content = element.text_content().strip()
            if (element.tag == 'p' and text_content.startswith('This borrower')):
                break

        lst_description = []
        for element in iter_elements:
            text_content = element.text_content().strip()
            if (element.tag == 'p' and text_content.startswith('A credit bureau reported the following information about')):
                text_date_credit_bureau = ' '.join(text_content.split(' ')[-3:])[:-1]
                table_key += ['date_credit_bureau']
                table_val += [text_date_credit_bureau]

                table_key += ['description']
                table_val += [' '.join(lst_description).strip()]
                break
            if (element.tag == 'p' and text_content != ''):
                lst_description.append(text_content.replace('\r\n', ' ').replace('\n', ' '))

        # find 3rd table
        for element in iter_elements:
            if element.tag == 'table':
                table_key += [x.text_content().strip() for x in element.xpath('.//tr/td[position() mod 2 = 1]')]
                table_val += [x.text_content().strip() for x in element.xpath('.//tr/td[position() mod 2 = 0]')]
                break
        
        for element in iter_elements:
            text_content = element.text_content().strip()
            if (element.tag == 'p' and text_content.startswith('The following answers to questions')):
                break
        for element in iter_elements:
            # find 4th table
            if element.tag == 'table':
                table_question = [x.text_content().strip() for x in element.xpath('.//td[1]')]
                table_answer = [x.text_content().strip() for x in element.xpath('.//td[2]')] 
                for i, (question, answer) in enumerate(zip(table_question[1:], table_answer[1:])):
                    table_key += [f'Question {i+1}', f'Answer {i+1}']
                    table_val += [question.replace("\r\n", " ").replace("\n", " "), answer.replace("\r\n", " ").replace("\n", " ")]
                break
        return key_val_to_dict(table_key, table_val)
    
    lst_values = []
    # for pos_start, pos_stop in zip(lst_positions, lst_positions[1:]):
    #     lst_values.append(get_values(iter(main_div[pos_start:pos_stop])))
    lst_iters = [iter(main_div[pos_start:pos_stop]) for pos_start, pos_stop in zip(lst_positions, lst_positions[1:])]
    with ThreadPool(processes=8) as tp:
        for result in tp.imap_unordered(get_values, lst_iters, chunksize=50):
            lst_values.append(result)

    df_detailed = pd.DataFrame(lst_values,)
    df_detailed.replace('n/a', pd.NA, inplace=True)
    
    lst_cols_dates = ['Sale and Original Issue Date', 'Initial maturity', 'Final maturity', 'date_requested', 'date_credit_bureau']
    for col in lst_cols_dates:
        if col in df_detailed.columns:
            df_detailed[col] = pd.to_datetime(df_detailed[col], format='%B %d, %Y').dt.date
        
    lst_cols_dollars = ['Aggregate principal amount of Notes offered', 
                       'Aggregate principal amount of Notes sold', 
                       'Amount of corresponding member loan funded by Lending Club',
                       'Delinquent Amount',
                       'Revolving Credit Balance']
    for col in lst_cols_dollars:
        if col in df_detailed.columns:
            df_detailed[col] = df_detailed[col].str.lstrip('$').str.replace(',', '').astype(float)
    
    lst_cols_percents = ['Stated interest rate', 'Service Charge', 'Debt-to-income ratio', 
                         'Joint Debt-to-Income',
                         'Revolving Line Utilization']
    for col in lst_cols_percents:
        if col in df_detailed.columns:
            df_detailed[col] = pd.to_numeric(df_detailed[col].str.rstrip('%').str.replace(',', ''), errors='coerce')
    
    # numerical variables
    lst_cols_int = ['Accounts Now Delinquent', 'Open Credit Lines', 'Delinquencies (Last 2 yrs)',
                    'Total Credit Lines', 'Months Since Last Delinquency',
                    'Public Records On File', 'Months Since Last Record',
                    'Inquiries in the Last 6 Months']
    for col in lst_cols_int:
       if col in df_detailed.columns:
            df_detailed[col] = pd.to_numeric(df_detailed[col], errors='coerce')
    
    # monthly to annual
    lst_cols_monthly = ['Gross income', 'Joint Gross Income']
    for col in lst_cols_monthly:
        if col in df_detailed.columns:
            df_detailed[col] = pd.to_numeric(df_detailed[col].apply(clear_monthly_salary), errors='coerce')
            df_detailed[f'{col} (annual)'] = df_detailed[col] * 12
    
    # info on prospectus and suplement
    lst_columns_order2 = list(df_detailed.columns)
    for k, v in dic_prospectus.items():
        df_detailed[k] = v
    
    lst_columns_order1 = list(dic_prospectus.keys())
    df_detailed = df_detailed[lst_columns_order1 + lst_columns_order2]

    excel_filepath = Path('data/external/lending-club/sales-reports-xlsx') / f'{html_filepath.stem}.xlsx'
    df_detailed.to_excel(excel_filepath, index=False)

def column_types_xlsx() -> dict:
    """
    Returns the list of column types read from the html reports
    """
    lst_cols_dates = ['Sale and Original Issue Date', 'Initial maturity', 'Final maturity', 'date_requested', 'date_credit_bureau']
    lst_cols_dollars = ['Aggregate principal amount of Notes offered', 
                       'Aggregate principal amount of Notes sold', 
                       'Amount of corresponding member loan funded by Lending Club',
                       'Delinquent Amount',
                       'Revolving Credit Balance']
    lst_cols_int = ['Accounts Now Delinquent', 'Open Credit Lines', 'Delinquencies (Last 2 yrs)',
                'Total Credit Lines', 'Months Since Last Delinquency',
                'Public Records On File', 'Months Since Last Record',
                'Inquiries in the Last 6 Months']
    lst_cols_monthly = ['Gross income', 'Joint Gross Income']
    
    dic_column_types_xlsx = {'dates': lst_cols_dates,
                             'dollars': lst_cols_dollars,
                             'int': lst_cols_int,
                             'monthly': lst_cols_monthly,
                             'annual' : [x + ' (annual)' for x in lst_cols_monthly]}
    return dic_column_types_xlsx

def colmns_dtypes_for_df() -> dict:
    """
    Returs a dictionary to read data from xlsx files
    """
    dic_types = column_types_xlsx()
    lst_types = [(x, str) for v in dic_types['dates'] for x in v]
    lst_types += [(x, float) for v in dic_types['dollars'] for x in v]
    lst_types += [(x, float) for v in dic_types['int'] for x in v]
    lst_types += [(x, float) for v in dic_types['monthly'] for x in v]
    lst_types += [(x, float) for v in dic_types['annual'] for x in v]
    return dict(lst_types)

def read_sales_report_excel():
    """
    Reads into one dataframe data from slessup reports xlsx files
    """
    excel_folderpath = Path('data/external/lending-club/sales-reports-xlsx')
    lst_excel_filepaths = list(excel_folderpath.glob('salessup_????????.xlsx'))
    lst_excel_filepaths = list(excel_folderpath.glob('salessup_2016????.xlsx'))
    
    dic_types_list = column_types_xlsx() 
    dic_types  = colmns_dtypes_for_df()
    lst_dfs = []
    for excel_filepath in tqdm(lst_excel_filepaths):
        df = pd.read_excel(excel_filepath, dtype=dic_types, engine='openpyxl')
        lst_dfs.append(df)

    lst_dfs = [x for x in lst_dfs if x.shape[0] > 0]
    df_combined_fragmented = pd.concat(lst_dfs, ignore_index=True, axis=0)
    df_combined = df_combined_fragmented.copy(deep=True)
    del df_combined_fragmented
    
    for col in dic_types_list['dates']:
        df_combined[col] = pd.to_datetime(df_combined[col])
    
    df_combined['Sale and Original Issue Date'].max()
    df_combined['year_quarter'] = df_combined['Sale and Original Issue Date'].dt.to_period('Q')
    df_combined.to_excel(excel_folderpath / 'combined_2016.xlsx')
    
    df_combined_2016Q1 = df_combined[df_combined['year_quarter'] == '2016Q1']
    
    df_data = zipped_read('LoanStats_2016Q1.csv.zip')
    
    df_data['emp_title'].value_counts()
    df_data.shape
    df_combined_2016Q1.shape
    
    ser_quarter_statistics = df_combined.groupby('year_quarter').size()
    ser_quarter_statistics.index = ser_quarter_statistics.index.astype(str)
    ser_quarter_statistics.to_excel(excel_folderpath / 'quarter_statistics.xlsx')
    df_combined_2016Q1['suplement_date'].value_counts()

    df_combined_2016Q1.to_excel(excel_folderpath / 'combined_2016Q1.xlsx')
    df_combined_2016Q1['suplement_no'].value_counts(sort=False).sort_index()
    cols_to_check1 = ['Debt-to-income ratio', 'Stated interest rate', 'Aggregate principal amount of Notes offered',  
                      'Open Credit Lines',  'Total Credit Lines']
    is_unique = not df_combined_2016Q1.duplicated(subset=cols_to_check1, keep=False).any()
    print(is_unique)
    duplicates = df_combined_2016Q1[df_combined_2016Q1.duplicated(subset=cols_to_check1, keep=False)]
    print(duplicates['Job title'])

    df_combined_2016Q1[cols_to_check].value_counts()

    lst_cols_percents = ['int_rate']
    for col in lst_cols_percents:
        if col in df_data.columns:
            df_data[col] = pd.to_numeric(df_data[col].str.rstrip('%').str.replace(',', ''), errors='coerce')
        
    cols_to_check2 = ['emp_title', 'dti', 'int_rate', 'loan_amnt',  'open_acc', 'total_acc']
    is_unique2 = not df_data.duplicated(subset=cols_to_check2, keep=False).any()
    print(is_unique2)
    duplicates = df_data[df_data.duplicated(subset=cols_to_check2, keep=False)]
    duplicates_t = duplicates.T
    duplicates_t[duplicates_t[2872] != duplicates_t[79662]].to_excel(excel_folderpath / 'duplicates.xlsx')
    
      
    
    df_data.loc[df_data['emp_title'].str.startswith('!st Vice President/Wealth Adviso', na=False)]
    df_combined_2016Q1.loc[df_combined_2016Q1['Job title'].str.startswith('!st Vice President/Wealth Adviso', na=False),'Total Credit Lines']

    merged_df = pd.merge(df_combined_2016Q1, df_data, 
                    left_on=cols_to_check1,
                    right_on=cols_to_check2,
                    how='outer', indicator='merge_status')
    merged_df.columns
    merged_df['merge_status'].value_counts()
    merged_df.to_excel(excel_folderpath / 'merged.xlsx')

def is_unique_combination(df, lst_cols):
    return not df.duplicated(subset=lst_cols, keep=False).any()


def compare_datasets():
    """
    Compares datastes that were created from prospects (df1) and that were published by
    the company with the performance status (df2)
    
    As an example only Q1 2016 is taken for a start
    """
    folder_tmp = Path('data/external/lending-club/temp')
    dic_types  = colmns_dtypes_for_df()
    # dataframe with data from html reports 
    df1 = df_combined
    df1 = pd.read_excel(folder_tmp / 'combined_2016Q1.xlsx', dtype=dic_types)
    lst_cols_unique_df1 =  ['Debt-to-income ratio', 'Stated interest rate', 'Aggregate principal amount of Notes offered',  
                      'Open Credit Lines',  'Total Credit Lines', 'Earliest Credit Line']
    df1['Earliest Credit Line'] = pd.to_datetime(df1['Earliest Credit Line'], format='%m/%Y')
    df1['Earliest Credit Line'] = df1['Earliest Credit Line'].dt.to_period('M')
    is_unique_combination(df1, lst_cols_unique_df1)
    df1 = df1.dropna(axis=1, how='all')
    df1.shape
    
    # dataframe from aggergated published values
    df2 = pd.read_excel(folder_tmp / 'LoanStats_2016Q1.csv.xlsx')
    lst_cols_unique_df2 = ['dti', 'int_rate', 'loan_amnt',  'open_acc', 'total_acc', 'earliest_cr_line'] 
    df2['earliest_cr_line'] = pd.to_datetime(df2['earliest_cr_line'], format='%b-%Y')
    df2['earliest_cr_line'] = df2['earliest_cr_line'].dt.to_period('M')
    df2['int_rate'] = pd.to_numeric(df2['int_rate'].str.rstrip('%').str.replace(',', ''), errors='coerce')
    df2 = df2.dropna(axis=1, how='all')
    is_unique_combination(df2, lst_cols_unique_df2)
    
    # merging data - outer, left and right
    merged_df = pd.merge(df1, df2, 
                left_on=lst_cols_unique_df1,
                right_on=lst_cols_unique_df2,
                how='outer', indicator='merge_status')
    merged_df['merge_status'].value_counts()
    merged_df[merged_df['merge_status'] == 'right_only'].to_excel(folder_tmp / 'merged_right_only.xlsx')
    merged_df[merged_df['merge_status'] == 'left_only'].to_excel(folder_tmp / 'merged_left_only.xlsx')
    merged_df[merged_df['merge_status'] == 'both'].to_excel(folder_tmp / 'merged_both.xlsx')
    
    merged_df.loc[merged_df['merge_status'] == 'both', ['Application Type','application_type']+
                                                       ['Stated interest rate', 'int_rate']+
                                                       ['Location','zip_code']]
    
    df2[df2['emp_title'].str.startswith('Patent Legal', na=False)]
    df1[df1['Job title'].str.startswith('Patent Legal', na=False)]
    df1.loc[489452,:]
    df1.set_index('Unnamed: 0', inplace=True)
    df1.at[489452,'Job title']    

    emp_title = 'MINISTER IN CHARGE'
    df1.loc[df1['Job title'].str.startswith(emp_title, na=False),'Job title']
    df2.loc[df2['emp_title'].str.startswith(emp_title, na=False), 'emp_title']
    merged_df.loc[merged_df['emp_title'].str.startswith(emp_title, na=False), 'emp_title']
    merged_df.loc[merged_df['Job title'].str.startswith(emp_title, na=False), 'Job title']
    merged_df.loc[[0, 90055],].T.to_excel(folder_tmp  / 'compare.xlsx')
    
    df2[df2['emp_title'].str.startswith('MINISTER IN CHARGE', na=False)].T.to_excel(folder_tmp / 'minister.xlsx')
    
    merged_df.loc[merged_df['merge_status'] == 'right_only', ['emp_title']+lst_cols_unique_df2].to_excel(folder_tmp  / 'only_right.xlsx')

    merged_df.groupby(['merge_status', 'initial_list_status']).size()

    df2['title']

def sales_report_from_folder():
    """
    Loops through files in the folder and creates Excel file from html
    """
    html_folderpath = Path('data/external/lending-club/sales-reports-htm')
    lst_html_filepaths = list(html_folderpath.glob('*.htm'))
    for html_filepath in tqdm(lst_html_filepaths):
        print(html_filepath.name)
        sales_report_from_html_to_excel(html_filepath)
    

def zipped_read(file=None):
    path_inp = Path('data/external/lending-club')
    if file is None:
        file = 'LoanStats3a.csv.zip'
    zip_file = path_inp / file
    df = pd.read_csv(zip_file, compression='zip', header=0, sep=',', 
                            quotechar='"', skip_blank_lines=True, 
                            skipfooter=2, skiprows=1, engine='python')
    
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df['issue_d'].value_counts(sort=False).sort_index()


    excel_filepath = Path('data/external/lending-club/temp') / f'{zip_file.stem}.xlsx'
    df.to_excel(excel_filepath)
    
    return df
        
def concatenate_lending_club():
    """
    Concatenates lines from files matching the input pattern that start with a double quote 
    into a single output file.

    Args:
        input_pattern: A glob pattern to match input files (e.g., "*.txt", "data/*.csv").
        output_file: Path to the output file.
    """
    path_out = Path('data/external/lending-club')
    output_file = path_out / 'LoanStats.csv'
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        lst_files = ['LoanStats3a.csv', 'LoanStats3b.csv', 'LoanStats3c.csv', 'LoanStats3d.csv']
        lst_files += [f'LoanStats_{year}Q{quarter}.csv' for year, quarter in product(range(2016, 2022), range(1, 5))]

        for file_path in lst_files:
            try:
                with open(path_out / file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()  # Remove leading/trailing whitespace
                        if line.startswith('"'):
                            outfile.write(line + '\n')  # Add newline after each line

            except Exception as e:
                print(f"Error processing file '{file_path}': {e}")

# Dictionary of types of columns used for reading the dataset into pandas
DIC_DTYPE = {'loan_amnt':float, 
                'term':'Int64', 
                'int_rate':float, 
                'grade': 'category', 
                'sub_grade': 'category', 
                'emp_title': str, 
                'emp_length': 'category', 
                'home_ownership': 'category', 
                'annual_inc':float, 
                'verification_status': 'category',
                'loan_status': 'category', 
                'desc': str,
                'purpose':'category',
                'title' : str, 
                'zip_code': str, 
                'addr_state': 'category',
                'dti':float, 
                'delinq_2yrs':'Int64', 
                'earliest_cr_line':str,
                'inq_last_6mths' : 'Int64', 
                'mths_since_last_delinq':'Int64',
                'mths_since_last_record' :'Int64', 
                'open_acc': 'Int64',
                'pub_rec':'Int64', 
                'revol_bal':float, 
                'revol_util':float,
                'total_acc':'Int64', 
                'acc_now_delinq':'Int64', 
                'delinq_amnt':float,
                'mths_since_last_major_derog':'Int64', 
                'application_type':'category',
                'annual_inc_joint':float, 
                'dti_joint':float, 
                'verification_status_joint':'category'}


def merge_lending_club_only_selected_columns():
    """
    Merge into a single output file with selected columns.

    Args:
        input_pattern: A glob pattern to match input files (e.g., "*.txt", "data/*.csv").
        output_file: Path to the output file.
    """
    path_inp = Path('data/external/lending-club/loan-stats-csv')
    path_out = Path('data/external/lending-club/loan-stats-merged')
    path_out.mkdir(parents=True, exist_ok=True)
    output_file = path_out / 'LoanStats_merged.csv'
    
    lst_loan_description = ['loan_status', 'issue_d', 'loan_amnt', 'term', 'int_rate', 'purpose', 'application_type']
    lst_loan_provided = ['emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 
                         'desc', 'title', 'zip_code', 'addr_state', 'dti', 'annual_inc_joint', 'dti_joint' , 'verification_status_joint'] 
    lst_credit_bureau = ['grade', 'sub_grade', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 
                         'mths_since_last_delinq', 'mths_since_last_record',
                         'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
                         'mths_since_last_major_derog', 'acc_now_delinq', 'delinq_amnt']
    lst_cols = lst_loan_description + lst_loan_provided + lst_credit_bureau

    lst_files = ['LoanStats3a.csv.zip', 'LoanStats3b.csv.zip', 'LoanStats3c.csv.zip', 'LoanStats3d.csv.zip']
    lst_files += [f'LoanStats_{year}Q{quarter}.csv.zip' for year, quarter in product(range(2016, 2021), range(1, 5))]

    # for columns with percents, temporarily it needs to be read as string
    dic_dtype = DIC_DTYPE.copy()
    lst_cols_percents = ['int_rate', 'revol_util']
    for col in lst_cols_percents:
        dic_dtype[col] = str

    df_merged = pd.DataFrame()
    for file in tqdm(lst_files):
        try:
            zip_file = path_inp / file
            df = pd.read_csv(zip_file, compression='zip', header=0, sep=',', 
                            quotechar='"', skip_blank_lines=True, 
                            skipfooter=2, skiprows=1, engine='python',
                            usecols = lst_cols,
                            dtype = dic_dtype)
            df_merged = pd.concat([df_merged, df], ignore_index=True, )
        except Exception as e:
            print(f"Error processing file '{file}': {e}")

    # simplify some columns
    df_merged['term'] = pd.to_numeric(df_merged['term'].str.rstrip(' months'), errors='coerce')

    lst_cols_percents = ['int_rate', 'revol_util']
    for col in lst_cols_percents:
        if col in df.columns:
            df_merged[col] = pd.to_numeric(df_merged[col].str.rstrip('%').str.replace(',', ''), errors='coerce')
    
    df_merged = df_merged[lst_cols]
    df_merged.to_csv(output_file, index=False)
    print(output_file)

def read_merged_file():
    path_inp = Path('data/external/lending-club/loan-stats-merged')
    file = path_inp / 'LoanStats_merged.csv'
    df = pd.read_csv(file, dtype=DIC_DTYPE)
    df['loan_status'].value_counts()
    df.groupby(['loan_status', 'issue_d']).size()
    counts = df.groupby(['loan_status', 'issue_d']).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y').dt.to_period('M')
    
    shares_pivot = df.pivot_table(index='issue_d', columns='loan_status', aggfunc='size', fill_value=0)
    shares_pivot = shares_pivot.apply(lambda x: round(x / x.sum() * 100,2), axis=1)
    shares_pivot.index = shares_pivot.index.astype(str)
    shares_pivot.to_excel(path_inp / 'LoanStats_pivot.xlsx', float_format="%.2f") 
    df.head(100).to_excel(path_inp / 'LoanStats_merged_head.xlsx', index=False)

def clean_merged_file():
    """
    Input file(s):  LoanSats_merged.csv
    Output file(s): LoanStats_merged_cleaned.csv, LoanStats_pivot_cleaned.xlsx
    
    1. Changes the lona status os that it is either Fully Paid or Charged Off.
    2. Calculates difference beteen issue date and earliest credit line, diff_issue_d_earliest_cr_line 
    """
    path_inp = Path('data/external/lending-club/loan-stats-merged')
    file = path_inp / 'LoanStats_merged.csv'
    df = pd.read_csv(file, dtype=DIC_DTYPE)
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y').dt.to_period('M')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y').dt.to_period('M')
    df['loan_status'].value_counts()
    df[['issue_d','term']]
    dic_status_map = {'Does not meet the credit policy. Status:Fully Paid' : 'Fully Paid', 
                      'Does not meet the credit policy. Status:Charged Off' : 'Charged Off',
                      'Default' : 'Charged Off'}
    df['loan_status'] = df['loan_status'].map(dic_status_map).fillna(df['loan_status'])
    df['loan_status'].value_counts()
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    
    def get_month_diff(period_delta):
        if pd.isna(period_delta):  # Check for NaTType
            return None  # Or return another value like -1, np.nan, or a string
        else:
            return period_delta.n
    df['diff_issue_d_earliest_cr_line'] = (df['issue_d'] - df['earliest_cr_line']).apply(get_month_diff)
    
    shares_pivot = df.pivot_table(index='issue_d', columns=['loan_status','term'], aggfunc='size', fill_value=0)
    shares_pivot = shares_pivot.apply(lambda x: round(x / x.sum() * 100,2), axis=1)
    shares_pivot.index = shares_pivot.index.astype(str)
    shares_pivot.to_excel(path_inp / 'LoanStats_pivot_cleaned.xlsx', float_format="%.2f") 
    # save to file 
    file2 = path_inp / 'LoanStats_merged_cleaned.csv'
    df.to_csv(file2, index=False)
    print(f'{file2} created')

def draw_stratfied_sample():
    """
    Input file(s):  LoanStats_merged_cleaned.csv, 
    Output file(s): LoanStats_merged_cleaned_sample.csv, LoanStats_merged_cleaned_sample.xlsx, 
    
    CSV and XLSX files are the ssmae
    
    """
    path_inp = Path('data/external/lending-club/loan-stats-merged')
    file = path_inp / 'LoanStats_merged_cleaned.csv'
    df = pd.read_csv(file, dtype=DIC_DTYPE)

    lst_category_columns = ['loan_status', 'issue_d', 'term']
    random_state = 42
    np.random.seed(random_state)
    df['random_number'] = np.random.rand(len(df))
    
    def sample_group(group):
        sample_percentage = 0.01
        group_size = max(1, int(len(group) * sample_percentage))
        return group.sort_values('random_number').head(group_size)
    
    sampled_df = df.groupby(lst_category_columns, group_keys=False, observed=True).apply(sample_group).copy()
    sampled_df = sampled_df.drop(columns=['random_number'])
    
    path_out = Path('data/external/lending-club/loan-stats-merged')
    file_out = path_out / 'LoanStats_merged_cleaned_sample.csv'
    sampled_df.to_csv(file_out, index=False)
    sampled_df.to_excel(file_out.with_suffix('.xlsx'), index=False)
    

def metadata_lending_club():
    """
    Input file(s):  LoanStats_merged_cleaned_sample.csv,
    Output file(s): lending_club.csv, lending_club_metadata.csv
    
    Procedure to change columns to numeric data, 
    so that it could be used in machine learning algorithms.
    
    In addition, it creates metadata file with information about the columns.
    """
    path_inp = Path('data/external/lending-club/loan-stats-merged')
    path_out= Path('data/processed/lending-club')
    path_out.mkdir(parents=True, exist_ok=True)
    
    file = path_inp / 'LoanStats_merged_cleaned_sample.csv'
    df = pd.read_csv(file, dtype=DIC_DTYPE |{'diff_issue_d_earliest_cr_line':'Int64'})
    df['target'] = df['loan_status'].map({'Fully Paid':1, 'Charged Off':0})
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%Y-%m').dt.to_period('M')
    df['issue_d_numeric'] = (df['issue_d'] - df['issue_d'].min()).apply(lambda x: x.n + 1)
    df['purpose'].value_counts()
    df['application_type'].value_counts(dropna=False, sort=False)
    df['emp_length'].unique().tolist()
    DIC_EMP_LENGTH = {'< 1 year':0, 
                      '1 year':1, '2 years':2,
                      '3 years':3, 
                      '4 years':4, '5 years':5, '6 years':6, 
                      '7 years':7, '8 years':8, 
                      '9 years':9,  '10+ years':10, }    
    df['emp_length_ordinal'] = df['emp_length'].map(DIC_EMP_LENGTH)    
    df['emp_length_ordinal'].value_counts(dropna=False)
    df['emp_length_ordinal'] = df['emp_length_ordinal'].astype('Int64')
    df['grade_ordinal'] = df['grade'].map({'A':35, 'B':30, 'C':25,'D':20, 'E':15, 'F':10, 'G':5})
      
    lst_subgrades = list(product(['A', 'B', 'C', 'D', 'E', 'F', 'G'], range(1,6)))
    max_subgrade = len(lst_subgrades)
    dic_subgrades = {f'{x[0]}{x[1]}':max_subgrade-n for n,x in enumerate(lst_subgrades, start=0)}
    df['sub_grade_ordinal'] = df['sub_grade'].map(dic_subgrades)

    df['verification_status_ordinal'] = df['verification_status'].map({'Not Verified':0, 'Source Verified':2, 'Verified':1})
    df['verification_status_joint_ordinal'] = df['verification_status_joint'].map({'Not Verified':0, 'Source Verified':2, 'Verified':1})
    # df['verification_status_ordinal'].value_counts()

    lst_cols_target = ['target']
    lst_cols_numerical = ['issue_d_numeric', 'loan_amnt', 'term', 'int_rate',
                          'emp_length_ordinal',  'annual_inc', 'dti', 'annual_inc_joint', 'dti_joint', 
                          'delinq_2yrs', 'inq_last_6mths', 
                          'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec',
                          'revol_bal', 'revol_util', 'total_acc', 'mths_since_last_major_derog', 
                          'acc_now_delinq', 'delinq_amnt', 'diff_issue_d_earliest_cr_line']
    lst_cols_ordinal = ['emp_length_ordinal', 'grade_ordinal', 'sub_grade_ordinal', 'verification_status_ordinal', ]
    lst_cols_categorical = ['purpose', 'application_type', 
                            'home_ownership', 
                            'zip_code', 
                            'addr_state',]
    lst_cols_text = ['emp_title', 'desc', 'title']
    lst_cols = lst_cols_target + lst_cols_numerical + lst_cols_categorical + lst_cols_text
    df = df[lst_cols]
    lst_cols_before = df.columns.to_list()
    df2 = pd.get_dummies(df, columns=lst_cols_categorical, dummy_na=True, drop_first=False, dtype='Int64')
    lst_cols_after = df2.columns.to_list()
    lst_cols_diff = [x for x in lst_cols_after if x not in lst_cols_before]
    df2[lst_cols_diff] = df2[lst_cols_diff].replace(0, np.nan)
    df2['addr_state_WA'].value_counts(dropna=False)
    df2.to_csv(path_out / 'lending_club.csv', index=False)
    # prepraring for lending_club_metadata.csv
    dic_metadata = dict()
    dic_metadata = dic_metadata | {x:'numerical' for x in lst_cols_numerical}
    dic_metadata = dic_metadata | {x:'ordinal' for x in lst_cols_ordinal}      
    dic_metadata = dic_metadata | {x:'categorical' for x in lst_cols_categorical}      
    dic_metadata = dic_metadata| {x:'text' for x in lst_cols_text}   
    df_metadata = pd.DataFrame.from_dict(dic_metadata, orient='index', columns=['type'])
    df_metadata.index.name = 'variable'
    df_metadata['description'] = ''
    df_metadata.to_csv(path_out / 'lending_club_metadata.csv', index=True)

       
def prepare_lending_club():
    """ 
    Here done in previous step
    """
    ...
    
    
def get_list_of_numerical(path_metadata_csv) -> list:
    """
    Returns the list of numerical variable for a given dataset. 
    """
    path_metadata_csv = Path(path_metadata_csv)     
    df_metadata = pd.read_csv(path_metadata_csv)
    lst_numerical = df_metadata.loc[df_metadata['type'] == 'numerical','variable'].to_list()
    return lst_numerical

get_list_of_numerical_lending_club = partial(get_list_of_numerical, 'data/processed/lending-club/lending_club_metadata.csv')

def stratified_k_folds_lending_club():
    """
    Prepares a file with the split into stratified 10-folds.
    Saves file japanese_folds.csv, where in each column there are observation
    numbers which will be used in test dataset in each fold. The observations
    for train subset are the remaining ones 
    """
    path_inp_dir = Path('data/processed/lending-club') 
    path_inp_csv = path_inp_dir / 'lending_club.csv'
    
    path_out_dir = Path('data/processed/lending-club') 
    path_out_csv = path_out_dir / 'lending_club_folds.csv'
    path_out_stats = path_out_dir / 'lending_club_stats.csv'
        
    df = pd.read_csv(path_inp_csv)
    # list of numerical variables:
    lst_numerical = get_list_of_numerical_lending_club()
    
    # folds
    df_shuffled = df.sample(frac=1, random_state=20)
    y = df_shuffled.pop('target')
    X = df_shuffled

    lst_original_iloc = df.index.to_list() 
    dic_orginal_iloc = {v:n for n,v in enumerate(lst_original_iloc)}
    lst_new_iloc = list(df_shuffled.index)
    
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    # StratifiedGroupKFold to be used here
    dic_combined = dict()
    dic_folds = dict()
    lst_stats_df = []
    for n_fold, (idx_train, idx_test) in enumerate(skf.split(X, y),start=1):
        dic_combined.update({x:n_fold for x in idx_test})
        dic_folds.update({f'fold{n_fold:02}':sorted([dic_orginal_iloc[lst_new_iloc[x]] for x in idx_test])})
        
        # Statistics min/max for training dataset 
        df_stats = pd.DataFrame()
        df_stats = X.loc[idx_train, lst_numerical].agg(['min','max'])
        df_stats['fold_num'] = n_fold   
        new_index = pd.MultiIndex.from_arrays([df_stats['fold_num'], df_stats.index], names=['fold_num', 'stats'])
        df_stats.index = new_index
        df_stats.drop('fold_num', axis=1, inplace=True)
        lst_stats_df.append(df_stats.copy())
        
    df_folds = pd.DataFrame.from_dict(dic_folds, orient='index')
    df_folds = df_folds.transpose()
    df_folds.to_csv(path_out_csv, index=False, float_format='%.0f')
    
    df_stats_agg = pd.concat(lst_stats_df)
    df_stats_agg.to_csv(path_out_stats, index=True)


def custom_format_float(value):
    """
    Custom format, so that no unnecessary numbers as saved into the file
    """
    if value == 0:
        return "0"  # Or any other representation you want for 0
    elif abs(value) < 1e-8: # handle very small numbers as 0
        return "0"
    else:
        return f"{value:.8f}".rstrip('0').rstrip('.') # removes trailing zeros and . if nothing left


def folds_to_csv_lending_club():
    """
    Converts data from the whole CSV dataset to csv split by folds.
    """
    # Read data
    path_inp_dir = Path('data/processed/lending-club') 
    path_inp_csv = path_inp_dir / 'lending_club.csv'
    df_data = pd.read_csv(path_inp_csv)
   
    # Reaad metadata    
    lst_numerical = get_list_of_numerical_lending_club()
        
    # Read folds
    path_inp_folds = path_inp_dir / 'lending_club_folds.csv'   
    df_folds = pd.read_csv(path_inp_folds)
    
    path_out_folds = Path('data/folds/lending-club')
    path_out_folds.mkdir(exist_ok=True, parents=True)
    for fold_n, fold_idx in df_folds.items():
        # Split data
        fold_idx_not_none = fold_idx.dropna()
        df_fold_test = df_data.iloc[fold_idx_not_none].copy()
        df_fold_train = df_data[~df_data.index.isin(df_fold_test.index)].copy()
        
        # Normalize data
        scaler = MinMaxScaler()
        df_fold_train[lst_numerical] = scaler.fit_transform(df_fold_train[lst_numerical])
        df_fold_test[lst_numerical] = scaler.transform(df_fold_test[lst_numerical])
        # Create csv files with folds
        df_fold_train.to_csv(path_out_folds.joinpath(f'{fold_n}_train.csv'), index=False, float_format=custom_format_float)
        df_fold_test.to_csv(path_out_folds.joinpath(f'{fold_n}_test.csv'), index=False, float_format=custom_format_float)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)
    # download_sales_lending_club()
    # df = read_lending_club()
    # extract_sales_repors_zip()
    # sales_report_from_folder()

    # merge_lending_club_only_selected_columns()
    # read_merged_file()
    # clean_merged_file()
    # draw_stratfied_sample()
    metadata_lending_club()