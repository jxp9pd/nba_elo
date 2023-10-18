"""
Functions supporting manual analysis of Kaizen Experiments

Imports & Constants:
    - metric lists (metrics for RHY targets and RHY core)
    - functions for hypothesis testing

Hypothesis Tests
z_test, t_test, and then functions that call them from the output of kaizen scorecard query.

Experiment Query Functions
All functions are able to handle segmentations by including a segment_table parameter. Functions
expect the segment_table to have schema | gdpr_user_uuid | segment |. Additional columns are fine.
    - get_exp_pop_query: returns a query that pulls all users and their segment if needed.
    - get_exp_across_days_query: returns a query that aggregates metrics to user level.
    - get_variant_metrics_query: returns a query that aggregates metrics to variant, segment level.
    - assemble_query: aggregates three above queries into one query.
    - experiment_main: Runs the full experiment pipeline. Takes in some input strings and returns a
      table of control, treatment, deltas, and p-value as a dataframe.
"""

# --------------------------------------------------------------------------------------------------
# Imports and Constants
# --------------------------------------------------------------------------------------------------

import scipy.stats
import numpy as np
import pandas as pd
import modular_visualizations as vis
from statsmodels.stats.proportion import proportions_ztest

TARGET_METRICS = ['has_rhy_transaction', 'rhy_transaction_volume', 'has_rhy_funded',
                  'rhy_transaction_count']
RHY_CORE = ['has_rhy', 'tapped_cash_tab', 'has_rhy_funded', 'rhy_deposit_amount',
            'rhy_transaction_count', 'rhy_transaction_volume', 'has_rhy_transaction',
            'has_declined_rhy_transaction', 'active_employer_direct_deposit_count',
            'has_received_employer_direct_deposit', 'card_dispute_percent']


# --------------------------------------------------------------------------------------------------
# Hypothesis Tests
# --------------------------------------------------------------------------------------------------


def two_proportion_z_test(data_dict):
    """
    Conducts a two proportion z test

    Parameters (data_dict)
    ----------
    p1: treatment proportion
    p2: control proportion
    n1: treatment pop. size
    n2: control pop. size
    """

    k1 = data_dict['p1'] * data_dict['n1']
    k2 = data_dict['p2'] * data_dict['n2']
    counts = np.array([k1, k2])
    nobs = np.array([data_dict['n1'], data_dict['n2']])
    stat, p_val = proportions_ztest(counts, nobs, alternative='two-sided')
    return p_val


def binary_metric_test(metric_name, df):
    """
    Returns the p-value of a diff. of proportions test

    Parameters
    ----------
    metric_name: original metric name
    df: experiment dataframe (already filtered to a segment if applicable)

    Assumes only two variants, 'member' and 'control'
    """
    col_name = f'avg_{metric_name}'
    z_test_dict = {
        'p1': df.loc[df.variant == 'member', col_name].item(),
        'p2': df.loc[df.variant == 'control', col_name].item(),
        'n1': df.loc[df.variant == 'member', 'pop_ct'].item(),
        'n2': df.loc[df.variant == 'control', 'pop_ct'].item()
    }
    return two_proportion_z_test(z_test_dict)


def two_sample_t_test(data_dict):
    """
    Conducts a difference of means t-test, returning the p-value.

    Parameters (data_dict)
    ----------
    mean1: treatment sample mean
    mean2: control sample mean
    var1: treatment sample variance
    var2: control sample variance
    n1: treatment pop. size
    n2: control pop. size
    """

    std1, std2 = np.sqrt(data_dict['var1']), np.sqrt(data_dict['var2'])
    t, p = scipy.stats.ttest_ind_from_stats(data_dict['mean1'], std1, data_dict['n1'],
                                            data_dict['mean2'], std2, data_dict['n2'])
    return p


def differences_of_means_test(metric_name, df):
    """
    Returns the p-value of a diff. of means t-test

    Parameters
    ----------
    metric_name: original metric name
    df: experiment dataframe (already filtered to a segment if needed)
    """

    avg_name = f'avg_{metric_name}'
    var_name = f'var_{metric_name}'
    t_test_dict = {
        'mean1': df.loc[df.variant == 'member', avg_name].item(),
        'mean2': df.loc[df.variant == 'control', avg_name].item(),
        'var1': df.loc[df.variant == 'member', var_name].item(),
        'var2': df.loc[df.variant == 'control', var_name].item(),
        'n1': df.loc[df.variant == 'member', 'pop_ct'].item(),
        'n2': df.loc[df.variant == 'control', 'pop_ct'].item()
    }
    return two_sample_t_test(t_test_dict)


# --------------------------------------------------------------------------------------------------
# Experiment Query Functions
# --------------------------------------------------------------------------------------------------

def get_exp_pop_query(data_dict):
    """
    Returns a query for experiment population.

    Parameters (data_dict)
    ----------
    exp_name: Kaizen experiment name
    segment_table: optional string of table with gdpr_user_uuid | segment |
    open_dt: string
    close_dt: string

    Note - in the event a user is listed in both control and member, this function will list them in
    member population. This function also does not exclude users from variant override.
    """

    base_query = '''
     WITH exp_pop AS
      (SELECT {segment_str}
              entity_id_mask AS gdpr_user_uuid,
              max(variant) AS variant,
              min(dt) AS dt
       FROM experiments.entity_exposures AS ee
       {segment_join}
       WHERE entity_type = 'user'
         AND experiment = '{experiment_name}'
         AND ee.dt BETWEEN '{open}' AND '{close}'
       GROUP BY {segment_str}
                ee.entity_id_mask)'''

    segment_str, segment_join = '', ''
    if 'segment_table' in data_dict:
        segment_str += 'segment,'
        segment_join += (
            f'LEFT JOIN hive.tmp.{data_dict["segment_table"]} AS s ON ee.entity_id_mask = s.gdpr_user_uuid')

    query_str = base_query.format(experiment_name=data_dict['exp_name'], open=data_dict['open_dt'],
                                  close=data_dict['close_dt'], segment_str=segment_str,
                                  segment_join=segment_join)
    return query_str


def get_user_exp_metrics(data_dict, client=None):
    """
    Takes a list of metrics and returns either string of the sql to select them or the query result
    depending on whether a client is passed.

    Parameters (data_dict)
    ----------
    metric_names: list of strings\n
    open_dt: string\n
    close_dt: string\n
    client: optional spark client

    """

    base_query = '''
    SELECT *
    FROM experiment_across_days
    '''

    exp_pop_str = get_exp_pop_query(data_dict)
    exp_across_days_str = get_exp_across_days_query(data_dict)
    query_str = exp_pop_str + ',' + exp_across_days_str + base_query
    if client is not None:
        return client.query(query_str)
    return query_str


def get_exp_across_days_query(data_dict):
    """
    Takes a list of metrics and returns a string of the sql to select them from the raw

    Parameters (data_dict)
    ----------
    metric_names: list of strings
    segment: optional boolean - assumed false
    open_dt: string
    close_dt: string
    """

    binary_str = '\t\t\tmax(coalesce(cast({metric_name} AS double), 0)) AS {metric_name},\n'
    num_str = '\t\t\tsum(coalesce(cast({metric_name} AS double), 0)) AS {metric_name},\n'
    select_str = '''
    experiment_across_days AS 
    (SELECT exp_pop.gdpr_user_uuid,{segment}
            variant,
{kaizen_cols}
    FROM experiments.user_metrics AS user_metrics
    JOIN exp_pop ON user_metrics.user_uuid_mask = exp_pop.gdpr_user_uuid
    AND user_metrics.dt >= exp_pop.dt{metric_table}
    WHERE user_metrics.dt BETWEEN '{open}' AND '{close}'
    GROUP BY exp_pop.gdpr_user_uuid,{segment}
             variant)'''
    segment_str, metric_table_join, kaizen_cols = '', '', ''
    if 'segment_table' in data_dict:
        segment_str += '\n\t\t\tsegment,'
    if 'custom_metric_names' in data_dict:
        metric_table_join += (
            f'\n\tLEFT JOIN hive.tmp.{data_dict["metric_table"]} AS m ON exp_pop.gdpr_user_uuid = m.gdpr_user_uuid')

    metric_cols = data_dict['metric_names'] + data_dict.get('custom_metric_names', [])
    for index, col in enumerate(metric_cols):
        if col[:3] == 'has':
            sql_str = binary_str.format(metric_name=col)
        else:
            sql_str = num_str.format(metric_name=col)
        if index == len(metric_cols) - 1:
            sql_str = sql_str[:-2]
        kaizen_cols += sql_str

    query_str = select_str.format(kaizen_cols=kaizen_cols, segment=segment_str, metric_table=metric_table_join,
                                  open=data_dict['open_dt'], close=data_dict['close_dt'])
    return query_str


def get_variant_metrics_query(data_dict):
    """
    Takes a list of metrics and summarize them for hypothesis tests
    Parameters (data_dict)
    ----------
    metric_names: list of strings
    segment: optional boolean - assumed false
    """

    base_query = '''
    SELECT {segment}
            variant,
            COUNT(gdpr_user_uuid) AS pop_ct,
    {kaizen_cols}
    FROM experiment_across_days
    GROUP BY {segment}
             variant
    ORDER BY {segment}
             variant'''

    segment_str = ''
    if 'segment_table' in data_dict:
        segment_str += 'segment,'

    metric_cols = data_dict['metric_names'] + data_dict.get('custom_metric_names', [])
    kaizen_cols = ''
    for index, col in enumerate(metric_cols):
        sql_str = ''
        if col[:3] != 'has':
            sql_str += f'\t\t\tvariance({col}) AS var_{col},\n'
        sql_str += f'\t\t\tavg({col}) AS avg_{col},\n'
        if index == len(metric_cols) - 1:
            sql_str = sql_str[:-2]
        kaizen_cols += sql_str
    query_str = base_query.format(kaizen_cols=kaizen_cols, segment=segment_str)
    return query_str


def assemble_query(data_dict):
    """
    Takes a full parameter dictionary and returns the final query string

    Parameters (data_dict)
    ----------
    metric_names: list of strings
    segment: optional boolean - assumed false
    open_dt: string
    close_dt: string
    """
    exp_pop_str = get_exp_pop_query(data_dict)
    exp_across_days_str = get_exp_across_days_query(data_dict)
    variant_query_str = get_variant_metrics_query(data_dict)

    return exp_pop_str + ',' + exp_across_days_str + variant_query_str


def experiment_main(data_dict, client):
    """
    Runs the full experiment pipeline

    Parameters (data_dict)
    ----------
    metric_names: list of strings of metrics.
    segment_table: table name in datalake of segmentations. Null value assumes no
                   segmentation is needed.
    open_dt: string format of YYYY-MM-DD. start date of experiment.
    close_dt: string format of YYYY-MM-DD end date of experiment.

    Return
    ------
    exp_df: variant, segment summarized
    exp_dict: table of results stored as dataframe
    """

    query_str = assemble_query(data_dict)
    exp_df = client.query(query_str)
    print('Query Complete')

    if 'segment' in exp_df.columns:
        segments = [exp_df[exp_df.segment == segment] for segment in exp_df.segment.unique().tolist()]
    else:
        segments = [exp_df]

    exp_results = []
    exp_dfs = []
    metric_cols = data_dict['metric_names'] + data_dict.get('custom_metric_names', [])
    for df in segments:
        p_vals = []
        for metric_name in metric_cols:
            # Hypothesis Tests
            if metric_name[:3] == 'has':
                p_val = binary_metric_test(metric_name, df)
                print(f'For {metric_name[4:]} the p-value is: {p_val}')
                p_vals.append(p_val)
            else:
                p_val = differences_of_means_test(metric_name, df)
                print(f'For {metric_name} the p-value is: {p_val}')
                p_vals.append(p_val)
        exp_dict = {'Metric': metric_cols, 'p_value': p_vals}

        # Data & Deltas
        exp_cols = ['avg_' + col for col in metric_cols]
        c_cols = df.loc[df.variant == 'control', exp_cols
        ].values.tolist()[0]
        t_cols = df.loc[df.variant == 'member', exp_cols
        ].values.tolist()[0]

        # Compile to dataframe
        exp_dict['Control'] = c_cols
        exp_dict['Treatment'] = t_cols
        exp_output = pd.DataFrame(exp_dict)
        exp_output['Delta'] = exp_output['Treatment'] - exp_output['Control']
        exp_output['Relative Delta'] = exp_output['Delta'] / exp_output['Control']
        exp_results.append(exp_output)
        exp_dfs.append(df)
    if len(exp_results) == 1:
        return exp_df, exp_results[0]
    return exp_dfs, exp_results

test_dict = {
    'metric_names': TARGET_METRICS,
    'exp_name': 'rhy-first-transaction-5-dollar-back',
    'custom_metric_names': ['merchant_reward_transaction_count', 'merchant_reward_transaction_volume',
                            'has_merchant_reward_transaction'],
    'metric_table': 'test_metrics_table_0717',
    'open_dt': '2023-06-16',
    'close_dt': '2023-06-17',
    'segment_table': 'incentive_v3_segments'
}

def metric_boxplot(df, metric_name, plot_dict=None):
    """
    Compares distribution between variants with a boxplot

    Parameters
    ----------
    df: user level dataframe with metric values and variant
    metric_name: string of metric name
    plot_dict: optional dict of plot parameters
    """

    if plot_dict is None:
        plot_dict = {
            'title': metric_name,
            'xlabel': 'Variant',
            'format_code': '{x:,.0f}'
        }
    plot_df = df[['variant', metric_name]]
    plot_df.columns = ['category', 'val']
    fig, ax = vis.boxplot_comparison(plot_df, plot_dict)
    return fig, ax

# result = get_user_exp_metrics(test_dict)
# print(result)

FINANCE_PATH = '/Users/john.pentakalos/Development/Data/X1/Finance/output/'
user_df = pd.read_csv(FINANCE_PATH + 'days_to_mobile_wallet_spend.csv')
user_df.rename(columns={'is_mobile_wallet_spender': 'variant'}, inplace=True)
