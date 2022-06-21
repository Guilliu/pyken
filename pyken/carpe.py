import os, statsmodels, statsmodels.api as sm, datetime
import numpy as np, pandas as pd

from scipy import special
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from IPython.display import display


####################################################################################################


def string_categories1(x, y):

    ''' Genera las categorías en una variable de tipo texto. Esto es
    un diccionario que mapea los n valores de la variable con los
    primeros n números naturales, en orden descendente por tasa de malos. '''

    return dict(map(reversed, enumerate(pd.Series(y).groupby(x, dropna=False)\
    .mean().sort_values(ascending=False).index.values)))


def string_categories2(breakpoints_cat):

    ''' Genera las categorías en una variable de tipo texto en función
    de una agrupación dada. Esto es un diccionario que mapea los n valores de
    la variable con los primeros n números naturales en el orden dela agrupación. '''

    try: return dict(map(reversed, enumerate([i for j in breakpoints_cat for i in j])))
    except: return {}


def string_to_num(x, categories):

    ''' Convierte un vector tipo texto a numérico transformando
    los valores según su mapeo en el diccionario categories. '''

    return pd.Series(x).map(categories).values


def num_to_string(x, categories):

    ''' Convierte un vector tipo numérico a texto transformando
    los valores según su mapeo en el diccionario categories. '''

    return pd.Series(x).map(dict((v, k) for k, v in categories.items())).values


def breakpoints_to_str(breakpoints_num, categories):

    ''' Convierte los breakpoints numéricos en los
    asociados de tipo texto siguiendo las categorias '''

    breakpoints_str = []
    for i in range(len(breakpoints_num)):
        if i == 0:
            breakpoints_str.append([j[0] for j in categories.items()
            if j[1] < breakpoints_num[i]])
        else:
            breakpoints_str.append([j[0] for j in categories.items()
            if breakpoints_num[i-1] <= j[1] < breakpoints_num[i]])
        if i == len(breakpoints_num) - 1:
            breakpoints_str.append([j[0] for j in categories.items()
            if breakpoints_num[i] <= j[1]])

    return breakpoints_str


def breakpoints_to_num(breakpoints_cat):

    if isinstance(breakpoints_cat[0], list):

        L, suma = [], 0
        for i in breakpoints_cat[:-1]:
            suma += len(i)
            L.append(suma-0.5)
        return np.array(L)

    else: return breakpoints_cat


def remapeo_missing(v, bp, old_value=-12345678):

    if isinstance(bp, dict):

        breakpoints = bp['breakpoints']
        missing_group = bp['missing_group']

        if missing_group!= 0:
            if missing_group == 1:
                return np.where(v == old_value, breakpoints[missing_group-1]-(np.e-2), v)
            if missing_group >= 2:
                return np.where(v == old_value, breakpoints[missing_group-2]+(np.e-2), v)
        else: return v
    else: return v


def compute_iv(x, y, breakpoints_num):

    ''' Calcula el iv de una columna numérica dado sus breakpoints. '''

    x_groups = np.digitize(x, breakpoints_num)

    ngroups = len(breakpoints_num) + 1
    g = np.empty(ngroups).astype(np.int64)
    b = np.empty(ngroups).astype(np.int64)

    for i in range(ngroups):

        g[i] = np.sum([(y == 0) & (x_groups == i)])
        b[i] = np.sum([(y == 1) & (x_groups == i)])

    total_g = g.sum()
    total_b = b.sum()

    pct_g = g / total_g
    pct_b = b / total_b

    iv = special.xlogy(pct_b - pct_g, pct_b / pct_g)

    return iv.sum()


def compute_group_names(dtype, breakpoints, missing_group=0, decimals=2):

    ''' Genera los nombres de los grupos de una columna numérica. '''

    if dtype == 'O': return breakpoints

    else:

        groups = np.concatenate([[-np.inf], breakpoints, [np.inf]])
        group_names1, group_names2 = [], []

        for i in range(len(groups) - 1):
            if np.isinf(groups[i]):
                a = '({0:.{2}f}, {1:.{2}f})'\
                .format(groups[i], groups[i+1], decimals)
            else: a = '[{0:.{2}f}, {1:.{2}f})'.format(groups[i], groups[i+1], decimals)
            group_names1.append(a)

        for group in group_names1:
            if '-12345670.00)' in group: group = 'Missing'
            if '[-12345670.00' in group: group = group.replace('[-12345670.00', '(-inf')
            group_names2.append(group)

        if missing_group != 0: group_names2[missing_group-1] += ', Missing'

        return group_names2


def compute_table(x, y, breakpoints_num, group_names, compute_totals=True):

    ''' Calcula la tabla de una columna numérica dado sus breakpoints. '''

    x_groups = np.digitize(x, breakpoints_num)

    ngroups = len(breakpoints_num) + 1
    g = np.empty(ngroups).astype(np.int64)
    b = np.empty(ngroups).astype(np.int64)

    for i in range(ngroups):

        g[i] = np.sum([(y == 0) & (x_groups == i)])
        b[i] = np.sum([(y == 1) & (x_groups == i)])

    e = g + b

    total_g = g.sum()
    total_b = b.sum()
    total_e = e.sum()

    pct_g = g / total_g
    pct_b = b / total_b
    pct_e = e / total_e

    b_rate = b / e
    woe = np.log(1 / b_rate - 1) + np.log(total_b / total_g)
    iv = special.xlogy(pct_b - pct_g, pct_b / pct_g)

    total_b_rate = total_b / total_e
    total_iv = iv.sum()

    table = pd.DataFrame({
        'Group': group_names,
        'Count': e,
        'Percent': pct_e,
        'Goods': g,
        'Bads': b,
        'Bad rate': b_rate,
        'WoE': woe,
        'IV': iv,
    })

    if compute_totals:
        table.loc['Totals'] = ['', total_e, 1, total_g,
        total_b, total_b_rate, '', total_iv]

    return table


def compute_scorecard(data, features, info, target_name='target', pvalues=False, redondeo=True):

    X = data.drop(target_name, axis=1).copy()
    y = data[target_name].values

    Xwoes = pd.DataFrame()
    scorecard, features_length = pd.DataFrame(), np.array([], 'int64')

    for feature in features:

        x = X[feature].values
        breakpoints_num = info[feature]['breakpoints_num']
        group_names = info[feature]['group_names']

        table = compute_table(x, y, breakpoints_num, group_names, False)
        table.insert(0, 'Variable', feature)
        scorecard = pd.concat([scorecard, table])
        features_length = np.append(features_length, len(table))

        Xwoes[feature] = transform_to_woes(x, y, breakpoints_num)

    log_reg = sm.Logit(y, sm.add_constant(Xwoes.values)).fit(method='lbfgs')
    coefs, intercept = np.array([log_reg.params[1:]]), np.array([log_reg.params[0]])

    scorecard['Raw score'] = scorecard['WoE'] * np.repeat(coefs.ravel(), features_length)
    scorecard['Aligned score'] = calib_score(scorecard['Raw score'], len(features), intercept)
    if redondeo: scorecard['Aligned score'] = scorecard['Aligned score'].round().astype('int')
    scorecard = scorecard.reset_index(drop=True)

    if pvalues: return scorecard, features_length, log_reg.pvalues
    else: return scorecard, features_length


def transform_to_woes(x, y, breakpoints_num):

    ''' Transforma a woes los valores de una columna numérica dado sus breakpoints. '''

    x_groups = np.digitize(x, breakpoints_num)

    ngroups = len(breakpoints_num) + 1
    g = np.empty(ngroups).astype(np.int64)
    b = np.empty(ngroups).astype(np.int64)

    for i in range(ngroups):

        g[i] = np.sum([(y == 0) & (x_groups == i)])
        b[i] = np.sum([(y == 1) & (x_groups == i)])

    e = g + b

    total_g = g.sum()
    total_b = b.sum()
    b_rate = b / e

    woe = np.log(1 / b_rate - 1) + np.log(total_b / total_g)

    mapeo_indices_woes = dict(zip([i for i in range(len(woe))], list(woe)))
    x_woes = pd.Series([mapeo_indices_woes[i] for i in x_groups])

    return x_woes


def calib_score(points, num_variables, intercept):

    n = num_variables
    pdo, odds, scorecard_points = 20, 1, 500

    factor = pdo / np.log(2)
    offset = scorecard_points - factor * np.log(odds)

    new_points = -(points + intercept / n) * factor + offset / n

    return new_points


def apply_scorecard(data, scorecard, info, id_columns=[], binary_prediction=True,
metrics=[], target_name='target', print_log=False, pre_text=''):

    features = list(scorecard['Variable'].unique())

    if metrics == []: data_final = data[id_columns + features].copy()
    else: data_final = data[id_columns + features + [target_name]].copy()
    data_final['scorecardpoints'] = 0.0

    for feature in features:

        x = data_final[feature]
        breakpoints_num = info[feature]['breakpoints_num']

        mapeo_points = scorecard[scorecard['Variable'] == feature]\
        .reset_index(drop=True)['Aligned score'].to_dict()

        data_final['scr_{}'.format(feature)] = \
        transform_to_points(x, breakpoints_num, mapeo_points)
        data_final['scorecardpoints'] += data_final['scr_{}'.format(feature)]

    if binary_prediction:
        data_final['prediction'] = np.where(data_final.scorecardpoints >= 500, 0, 1)

    columnas = list(data_final.columns).copy()
    columnas.remove('scorecardpoints')
    if binary_prediction: columnas.remove('prediction')
    columnas += ['scorecardpoints']
    if binary_prediction: columnas += ['prediction']
    data_final = data_final[columnas]

    if metrics == []: return data_final

    else:

        if metrics not in (['ks'], ['gini'], ['ks', 'gini'], ['gini', 'ks']):
            raise ValueError("Valor erroneo para 'metrics'. Los valores "
            "válidos son: ['ks'], ['gini'], ['ks', 'gini'], ['gini', 'ks']")

        if 'ks' in metrics:
            g = data_final.loc[data_final[target_name] == 0, 'scorecardpoints']
            b = data_final.loc[data_final[target_name] == 1, 'scorecardpoints']
            ks = ks_2samp(g, b)[0]

        if 'gini' in metrics:
            gini = 2*(1 - roc_auc_score(data_final[target_name],
            data_final['scorecardpoints'])) - 1

        if metrics == ['ks']:
            if print_log:
                print(pre_text + 'El modelo tiene un {:.2f}% de KS '
                'en esta muestra'.format(round(ks*100, 2)))
            return data_final, ks

        if metrics == ['gini']:
            if print_log:
                print(pre_text + 'El modelo tiene un {:.2f}% de Gini '
                'en esta muestra'.format(round(gini*100, 2), ))
            return data_final, gini

        if metrics in (['ks', 'gini'], ['gini', 'ks']):
            if print_log:
                print(pre_text + 'El modelo tiene un {:.2f}% de KS y un {:.2f}% de Gini '
                'en esta muestra'.format(round(ks*100, 2), round(gini*100, 2)))
            return data_final, ks, gini


def transform_to_points(x, breakpoints_num, mapeo_points):

    return pd.Series([mapeo_points[i] for i in np.digitize(x, breakpoints_num)])


def compute_final_breakpoints(variables, objetos, user_breakpoints):

    final_breakpoints = {}

    for variable in variables:
        try: final_breakpoints[variable] = user_breakpoints[variable]
        except: final_breakpoints[variable] = objetos[variable].breakpoints

    return final_breakpoints


def compute_info(X, variables, breakpoints):
    
    info = {}
    for variable in variables:
        
            info[variable] = {}
            bp = breakpoints[variable]
            
            if not isinstance(bp, dict):
                info[variable]['breakpoints_num'] = breakpoints_to_num(bp)
                info[variable]['group_names'] = compute_group_names(X[variable].values.dtype, bp)
                
            else:
                info[variable]['breakpoints_num'] = breakpoints_to_num(bp['breakpoints'])
                info[variable]['group_names'] = compute_group_names(
                X[variable].values.dtype, bp['breakpoints'], bp['missing_group'])
                
    return info


def features_selection(data, features, var_list, info, target_name='target',
method='stepwise', metric='pvalue', threshold=0.01, max_iters=12,
included_vars=[], muestra_test=None, show='gini', pre_text=''):
    
    N = 150

    if features != []: included_vars, max_iters = features, 0

    if method not in ('forward', 'stepwise'):
        raise ValueError(pre_text + "Valor inválido para el parámetro 'method'. "
        "Solo están pertimidos los valores 'forward' y 'stepwise'")

    if metric not in ('pvalue', 'ks', 'gini'):
        raise ValueError(pre_text + "Valor inválido para el parámetro 'metric'. "
        "Solo están pertimidos los valores 'pvalue', 'ks' y 'gini")

    if max_iters > len(var_list):
        print(pre_text + 'Cuidado, has puesto un valor numero máximo de iteraciones ({})'
        ' superior al número de variables candidatas ({})'.format(max_iters, len(var_list)))
        print(pre_text + '-' * N)
        max_iters = len(var_list)

    features = []

    num_included = len(included_vars)
    for i in range(num_included + max_iters):

        if i < num_included:

            new_var = included_vars.pop(0)
            features.append(new_var)

            if metric == 'pvalue':

                scorecard, features_length, pvalues = compute_scorecard(
                data, features, info, target_name=target_name, pvalues=True)
                train_final, ks_train, gini_train = apply_scorecard(
                data, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)
                if not isinstance(muestra_test, type(None)):
                    test_final, ks_test, gini_test = apply_scorecard(
                    muestra_test, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)

                if isinstance(muestra_test, type(None)):
                    if show == 'ks':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'KS train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1], ks_train*100, var))
                    if show == 'gini':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1], gini_train*100, var))
                    if show == 'both':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'KS train = {:.2f}% | Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1],
                        ks_train*100, gini_train*100, new_var))

                else:
                    if show == 'ks':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1], ks_train*100, ks_test*100, new_var))
                    if show == 'gini':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'Gini train = {:.2f}% | Gini test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1],
                        gini_train*100, gini_test*100, new_var))
                    if show == 'both':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% | Gini train = {:.2f}% | Gini test '
                        '= {:.2f}% ---> Feature selected: {}'.format(str(i+1).zfill(2), pvalues[-1],
                        ks_train*100, ks_test*100, gini_train*100, gini_test*100, new_var))


            else:

                scorecard, features_length = compute_scorecard(
                data, features, info, target_name=target_name)
                train_final, ks_train, gini_train = apply_scorecard(
                data, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)
                if not isinstance(muestra_test, type(None)):
                    test_final, ks_test, gini_test = apply_scorecard(
                    muestra_test, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)

                if isinstance(muestra_test, type(None)):
                    if show == 'ks':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | KS train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), ks_train*100, new_var))
                    if show == 'gini':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | Gini train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), gini_train*100, new_var))
                    if show == 'both':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | KS train = {:.2f}% | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), ks_train*100, gini_train*100, new_var))
                else:
                    if show == 'ks':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | KS train = {:.2f}% | '
                        'KS test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), ks_train*100, ks_test*100, new_var))
                    if show == 'gini':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | Gini train = {:.2f}% | '
                        'Gini test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), gini_train*100, gini_test*100, new_var))
                    if show == 'both':
                        print(pre_text + 'Step {} | Time - 0:00:00.000000 | KS train = {:.2f}% | '
                        'KS test = {:.2f}% | Gini train = {:.2f}% | Gini test = {:.2f}% '
                        '---> Feature selected: {}'.format(str(i+1).zfill(2), ks_train*100,
                        ks_test*100, gini_train*100, gini_test*100, new_var))

        else:

            if method == 'forward':

                if metric not in ('ks', 'gini'):
                    raise ValueError(pre_text + "El método 'forward' "
                    "solo se puede usar con las métricas 'ks' o 'gini'")

                start = datetime.datetime.now()

                contador = 0
                aux = pd.DataFrame(columns=['var', 'metric'])
                
                variables_excepcion_hessian = []
                for var in var_list:

                    if var not in features:

                        features.append(var)
                        try:
                            scorecard, features_length = compute_scorecard(
                            data, features, info, target_name=target_name)
                            data_final, metrica = apply_scorecard(
                            data, scorecard, info, metrics=[metric], target_name=target_name)
                            aux.loc[contador] = [var, metrica]
                        except statsmodels.tools.sm_exceptions.HessianInversionWarning as e:
                            variables_excepcion_hessian.append(variable)
                            variables_excepcion_hessian = list(set(variables_excepcion_hessian))
                        features.pop()
                        contador += 1

                aux = aux.sort_values('metric', ascending=False)
                new_var = aux.iloc[0]['var']
                features.append(new_var)

                scorecard, features_length = compute_scorecard(
                data, features, info, target_name=target_name)
                train_final, ks_train, gini_train = apply_scorecard(
                data, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)
                if not isinstance(muestra_test, type(None)):
                    test_final, ks_test, gini_test = apply_scorecard(
                    muestra_test, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)

                if isinstance(muestra_test, type(None)):
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print(pre_text + 'Step {} | Time - {} | KS train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, ks_train*100, new_var))
                    if show == 'gini':
                        print(pre_text + 'Step {} | Time - {} | Gini train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, gini_train*100, new_var))
                    if show == 'both':
                        print(pre_text + 'Step {} | Time - {} | KS train = {:.2f}% | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, ks_train*100, gini_train*100, new_var))
                else:
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print(pre_text + 'Step {} | Time - {} | KS train = {:.2f}% | '
                        'KS test = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, ks_train*100, ks_test*100, new_var))
                    if show == 'gini':
                        print(pre_text + 'Step {} | Time - {} | Gini train = {:.2f}% | '
                        'Gini test = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, gini_train*100, gini_test*100, new_var))
                    if show == 'both':
                        print(pre_text + 'Step {} | Time - {} | KS train = {:.2f}% | '
                        'KS test = {:.2f}% | Gini train = {:.2f}% | Gini test = {:.2f}% '
                        '---> Feature selected: {}'.format(str(i+1).zfill(2), end - start,
                        ks_train*100, ks_test*100, gini_train*100, gini_test*100, new_var))

            elif method == 'stepwise':

                if metric != 'pvalue':
                    raise ValueError(pre_text + "El método 'stepwise' "
                    "solo se puede usar con la métrica 'pvalue'")

                start = datetime.datetime.now()

                contador = 0
                aux = pd.DataFrame(columns=['var', 'pvalue'])
                
                variables_excepcion_hessian = []
                for var in var_list:

                    if var not in features:

                        features.append(var)
                        try:
                            scorecard, features_length, pvalues = compute_scorecard(
                            data, features, info, target_name=target_name, pvalues=True)
                            pvalue = pvalues[-1]
                            aux.loc[contador] = [var, pvalue]
                        except statsmodels.tools.sm_exceptions.HessianInversionWarning as e:
                            variables_excepcion_hessian.append(variable)
                            variables_excepcion_hessian = list(set(variables_excepcion_hessian))
                        features.pop()
                        contador += 1

                aux = aux.sort_values('pvalue')
                best_pvalue = aux.iloc[0]['pvalue']

                if best_pvalue >= threshold:
                    print(pre_text + '-' * N)
                    print(pre_text + 'Ya ninguna variable tiene un p-valor'
                    ' < {}, detenemos el proceso.'.format(threshold))
                    break

                new_var = aux.iloc[0]['var']
                features.append(new_var)

                scorecard, features_length, pvalues = compute_scorecard(
                data, features, info, target_name=target_name, pvalues=True)
                new_pvalue = pvalues[-1]
                train_final, ks_train, gini_train = apply_scorecard(
                data, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)
                if not isinstance(muestra_test, type(None)):
                    test_final, ks_test, gini_test = apply_scorecard(
                    muestra_test, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)

                if isinstance(muestra_test, type(None)):
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print(pre_text + 'Step {} | Time - {} | p-value = {:.2e} | '
                        'KS train = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, new_pvalue, ks_train*100, new_var))
                    if show == 'gini':
                        print(pre_text + 'Step {} | Time - {} | p-value = {:.2e} | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, new_pvalue, gini_train*100, new_var))
                    if show == 'both':
                        print(pre_text + 'Step {} | Time - {} | p-value = {:.2e} | '
                        'KS train = {:.2f}% | Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), end - start, new_pvalue,
                        ks_train*100, gini_train*100, new_var))

                else:
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print(pre_text + 'Step {} | Time - {} | p-value = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), end - start, new_pvalue,
                        ks_train*100, ks_test*100, new_var))
                    if show == 'gini':
                        print(pre_text + 'Step {} | Time - {} | p-value = {:.2e} | '
                        'Gini train = {:.2f}% | Gini test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), end - start, new_pvalue,
                        gini_train*100, gini_test*100, new_var))
                    if show == 'both':
                        print(pre_text + 'Step {} | Time - {} | p-value = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% | Gini train = {:.2f}% | Gini test '
                        '= {:.2f}% ---> Feature selected: {}'.format(str(i+1).zfill(2),
                        end - start, new_pvalue, ks_train*100, ks_test*100,
                        gini_train*100, gini_test*100, new_var))

                dict_pvalues = dict(zip(features, pvalues[1:]))
                to_delete = {}
                for v in dict_pvalues:
                    if dict_pvalues[v] >= threshold:
                        to_delete[v] = dict_pvalues[v]
                if to_delete != {}:
                    for v in to_delete:
                        features.remove(v)
                        scorecard, features_length = compute_scorecard(
                        data, features, info, target_name=target_name)
                        train_final, ks_train, gini_train = apply_scorecard(
                        data, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)
                        if not isinstance(muestra_test, type(None)):
                            test_final, ks_test, gini_test = apply_scorecard(
                            muestra_test, scorecard, info,
                            metrics=['ks', 'gini'], target_name=target_name)

                            if isinstance(muestra_test, type(None)):
                                end = datetime.datetime.now()
                                if show == 'ks':
                                    print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | KS train = {:.2f}% ---> Feature deleted : {}'\
                                    .format(str(i+1).zfill(2), dict_pvalues[v], ks_train*100, v))
                                if show == 'gini':
                                    print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | Gini train = {:.2f}% ---> Feature deleted : {}'\
                                    .format(str(i+1).zfill(2), dict_pvalues[v], gini_train*100, v))
                                if show == 'both':
                                    print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | KS train = {:.2f}% | Gini train '
                                    '= {:.2f}% ---> Feature deleted : {}'.format(str(i+1).zfill(2),
                                    dict_pvalues[v], ks_train*100, gini_train*100, v))

                            else:
                                end = datetime.datetime.now()
                                if show == 'ks':
                                    print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | KS train = {:.2f}% | KS test = {:.2f}% '
                                    '---> Feature deleted : {}'.format(str(i+1).zfill(2),
                                    dict_pvalues[v], ks_train*100, ks_test*100, v))
                                if show == 'gini':
                                    print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | Gini train = {:.2f}% | Gini test = {:.2f}% '
                                    '---> Feature deleted : {}'.format(str(i+1).zfill(2),
                                    dict_pvalues[v], gini_train*100, gini_test*100, v))
                                if show == 'both':
                                    print(pre_text + 'Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | KS train = {:.2f}% | KS test = {:.2f}% | '
                                    'Gini train = {:.2f}% | Gini test = {:.2f}% ---> Feature '
                                    'deleted : {}'.format(str(i+1).zfill(2), dict_pvalues[v],
                                    ks_train*100, ks_test*100, gini_train*100, gini_test*100, v))

    print(pre_text + '-' * N)
    print(pre_text + 'Selección terminada: {}'.format(features))
    print(pre_text + '-' * N)

    return features


def pretty_scorecard(modelo, color1='blue', color2='#FFFFFF'):
    
    if color1 == 'green': color1 = '#CCFFCC'
    if color1 == 'light_blue': color1 = '#CCFFFF'
    if color1 == 'blue': color1 = '#CCECFF'
    if color1 == 'pink': color1 = '#FFCCFF'
    if color1 == 'red': color1 = '#FFCCCC'
    if color1 == 'yellow': color1 = '#FFFFCC'
    if color1 == 'purple': color1 = '#CCCCFE'
    if color1 == 'orange': color1 = '#FFCC99'

    contador1, contador2, indices1, indices2 =  0, 0, [], []
    for i in modelo.features_length:
        for j in range(i):
            if contador1 % 2 == 0: indices1.append(contador2+j)
            else: indices2.append(contador2+j)
        contador1, contador2 = contador1+1, contador2+i

    def row_style(row):
        if row.name in indices1: return pd.Series('background-color: {}'.format(color1), row.index)
        else: return pd.Series('background-color: {}'.format(color2), row.index)

    try: display(modelo.scorecard.style.apply(row_style, axis=1))
    except: display(modelo.scorecard)


def parceling(df, breakpoints=[], tramos=15, id_columns=['id'],
score_name='scorecardpoints_acep', target_name='target', randomly=True):

    if randomly: np.random.seed(123)

    if breakpoints == []:

        tabla = proc_freq(df, score_name)

        inf = min(tabla.index)
        sup = max(tabla.index)
        salto = (sup - inf) / tramos
        breakpoints = [round(inf+i*salto, 2) for i in range(tramos)]

    print('Breakpoints:', breakpoints)

    df['parcel'] = np.digitize(df[score_name], breakpoints)
    a = proc_freq(df, 'parcel', target_name)
    a.columns.name = None
    a = a.reset_index(drop=True)
    a.index.name = 'parcel'
    b = proc_freq(df[df[target_name].isin([0, 1])], 'parcel', target_name, option='pct_row')
    b.columns.name = None
    b = b.reset_index(drop=True)
    b.index.name = 'parcel'
    b = b.rename(columns={0: '0_pct', 1: '1_pct'})
    c = a.merge(b, on='parcel', how='left')
    contador = 0
    molde = pd.DataFrame()
    for i in c.index:
        Xaux = df[(df['parcel'] == i+1) & (df['decision'] == 'rechazado')].copy()
        mascaritaaa = np.array([True]*round(len(Xaux)*c.loc[i]['1_pct'])
        +[False]*(len(Xaux)-round(len(Xaux)*c.loc[i]['1_pct'])))
        if randomly: np.random.shuffle(mascaritaaa)
        else: Xaux = Xaux.sort_values(score_name)
        Xaux['target_inf'] = np.where(mascaritaaa, 1, 0)
        contador += len(Xaux)
        molde = pd.concat([molde, Xaux])
    df2 = df.merge(molde[id_columns + ['target_inf']], how='left', on=id_columns)
    df2['target_def'] = np.where(df2['target_inf'].isna(), df2[target_name], df2['target_inf'])
    
    return df2, c


def cell_style(cell, name='Calibri', size=11, bold=False, italic=False, underline='none',
font_color='FF000000', background_color='', all_borders=False, hor_alignment='general',
ver_alignment='bottom', wrap_text=False, left_border=None, right_border=None, top_border=None,
bottom_border=None, left_border_color='FF000000', right_border_color='FF000000',
top_border_color='FF000000', bottom_border_color='FF000000'):
    
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    if background_color != '':
         fill_type = 'solid'
    else:
        background_color = 'FF000000'
        fill_type = None

    if all_borders == True:
        left_border, right_border, top_border, bottom_border = 'thin', 'thin', 'thin', 'thin'

    cell.font = Font(name=name, size=size, bold=bold,
    italic=italic, underline=underline, color=font_color)
    cell.fill = PatternFill(fill_type=fill_type, fgColor=background_color)
    cell.alignment = Alignment(horizontal=hor_alignment,
    vertical=ver_alignment, wrap_text=wrap_text)
    cell.border = Border(left=Side(border_style=left_border, color=left_border_color),
    right=Side(border_style=right_border, color=right_border_color),
    top=Side(border_style=top_border, color=top_border_color),
    bottom=Side(border_style=bottom_border, color=bottom_border_color))


def mini_log(ruta_log):

    L = ruta_log.split('/')
    L[-1] = 'mini_' + L[-1]
    nueva_ruta_log = '/'.join(L)

    f = open('{}'.format(nueva_ruta_log), 'w')

    with open('{}'.format(ruta_log)) as f:
        contents = f.readlines()

    with open('{}'.format(nueva_ruta_log), 'w') as f:
        for i in contents:
            if '123: ' in i:
                f.write(i[5:])
                f.write('\n')


def f_pd_read_hdfs_file(file_path, file_format='csv', edge_path='./', read_options={}):
    
    import subprocess

    origen, final = file_path, edge_path
    subprocess.call(['hdfs', 'dfs', '-copyToLocal', origen, final])

    egde_name = edge_path + file_path.split('/')[-1]

    options = {'filepath_or_buffer': egde_name}
    options.update(**read_options)

    if file_format == 'csv': df = pd.read_csv(**options)
    elif file_format == 'excel': df = pd.read_excel(**options)
    elif file_format == 'pickle': df = pd.read_pickle(**options)

    os.remove(egde_name)

    return df


def f_read_hdfs_model(hdfs_path, model_name, edge_path='.'):
    
    import subprocess, _pickle

    origen, final = '{}/{}.pckl'.format(hdfs_path, model_name), edge_path
    subprocess.call(['hdfs', 'dfs', '-copyToLocal', origen, final])

    pickle_file = open('{}/{}.pckl'.format(edge_path, model_name), 'rb')
    modelo = _pickle.load(pickle_file)
    pickle_file.close()

    os.remove('{}/{}.pckl'.format(edge_path, model_name))

    return modelo


def check_if_model_exists(hdfs_path, model_name):
    
    import subprocess

    return subprocess.call(['hdfs', 'dfs', '-test',
    '-e', '{}/{}.pckl'.format(hdfs_path, model_name)])


def calcula_ks_pyspark(df, target, score):
    
    import pyspark.sql.functions as sf
    from pyspark.ml.feature import Bucketizer

    df = df.withColumn(score, sf.round(sf.col(score), 3))
    minimo = df.agg({score: 'min'}).collect()[0][0]
    maximo = df.agg({score: 'max'}).collect()[0][0]
    bins = np.arange(minimo, maximo + 0.001, np.round((maximo - minimo) / 1000, 3))
    bins[0] = -float('inf')
    bins[len(bins) -1 ] = float('inf')

    bucketizer = Bucketizer(splits=list(bins), inputCol=score, outputCol='buckets')
    bucketed = bucketizer.setHandleInvalid('keep').transform(df)
    pre_pivot = bucketed.groupby('buckets', target).count().toPandas()
    pivot_table = pre_pivot.pivot_table(values='count', columns=target, index='buckets').fillna(0)
    pivot_table['pct_ceros'] = pivot_table.iloc[:, 0] / np.sum(pivot_table.iloc[:, 0].values)
    pivot_table['pct_unos'] = pivot_table.iloc[:, 1] / np.sum(pivot_table.iloc[:, 1].values)
    pivot_table['pct_ceros_cum'] = pivot_table['pct_ceros'].cumsum()
    pivot_table['pct_unos_cum'] = pivot_table['pct_unos'].cumsum()
    pivot_table['KS'] = (pivot_table['pct_ceros_cum'] - pivot_table['pct_unos_cum']).abs()
    KS = pivot_table['KS'].max()

    return pivot_table, KS


def compute_pyspark_formula(modelo):
    
    import copy

    pyspark_formula = []

    for i in modelo.features:

        aux = 'CASE '
        points = list(modelo.scorecard[modelo.scorecard['Variable'] == i]['Aligned score'])
        groups = copy.deepcopy(list(modelo.scorecard[modelo.scorecard['Variable'] == i]['Group']))

        for j in range(len(groups)):

            if modelo.objetos[i].dtype != 'O':
                if 'Missing' in groups[j]:
                    aux += 'WHEN (isnan({}) OR ({} IS NULL)) THEN {} '.format(i, i, points[j])
                if 'inf)' not in groups[j]:
                    lim = groups[j].split(', ')[-1][:-1]
                    if 'ss' in lim:
                        try: lim = groups[j].split(', ')[1][:-1]
                        except: continue
                    aux += 'WHEN {} < {} THEN {} '.format(i, lim, points[j])
                if 'inf)' in groups[j]:
                    lim = groups[j].split(', ')[0][1:]
                    if 'ss' not in lim:
                        aux += 'WHEN {} >= {} THEN {} '.format(i, lim, points[j])

            else:
                if 'Missing' in groups[j]:
                    aux += 'WHEN (isnan({}) OR ({} IS NULL)) THEN {} '.format(i, i, points[j])
                try: groups[j].remove('Missing')
                except: pass
                if groups[j] != []: aux += 'WHEN {} IN {} THEN {} '.format(i, groups[j], points[j])

        aux += 'ELSE {} END'.format(min(points))
        aux = aux.replace('[', '(').replace(']', ')')

        pyspark_formula.append(aux)

    return pyspark_formula


def proc_freq(data, row, col='', weight='', decimals=None, cumulative=False,
sort_col='', sort_dir='', option='', values=[], output=None):

    '''
    Generates the frequency table of a variable in a DataFrame. If two variables are passed,
    inside the 'row' and 'col' parameters, then it computes their crosstab.
    :param data: DataFrame. Table to use. Supports both pandas and spark Dataframe.
    :param row: str. Column to compute its frequency table.
    :param col: str. Column to compute its crosstab combined with 'row'.
    :param weight: str. Column with the frequencies of the distinct 'row' values.
    :param decimals: int. Decimal precision. Not rounded by default.
    :param sort_col: str. Column to sort by. It's sorted ascending on index by default.
    :param sort_dir: str. Direction to sort by. Use 'desc' for descending order.
    :param cumulative: bool. If True then returns cumulative frequency and percentage.
    :param option: str. By default, the crosstabs are computed with frequencies.
    Use 'pct_row' or 'pct_col' to obtain the desire percentages in crosstabs.
    :param values: list. In a frequency table as a pandas.DataFrame,
    it shows all the values of the list filling the ones that do not appear with zeros.
    :param output: SparkSession. By default the function returns a pandas.DataFrame.
    Input your spark session if a spark.DataFrame is wanted.
    :return:
    '''

    if type(data) == type(pd.DataFrame([])): # pandas.DataFrame

        if col == '': # Frequency table

            if weight == '': freq = data.groupby(row, dropna=False).size().to_frame()
            else: freq = data.groupby(row, dropna=False).agg({weight: 'sum'})
            freq.columns = ['frequency']

            if decimals == None: freq['percent'] = freq['frequency'] / freq['frequency'].sum()
            else: freq['percent'] = (freq['frequency'] / freq['frequency'].sum()).round(decimals)

            if sort_col == '' or sort_col == row:
                if sort_dir == 'desc': freq = freq.sort_index(ascending=False)
            else:
                if sort_dir == 'desc': freq = freq.sort_values(sort_col, ascending=False)
                else: freq = freq.sort_values(sort_col)

            if cumulative == True:
                freq['cumulative_frequency'] = freq['frequency'].cumsum()
                if decimals == None:
                    freq['cumulative_percent'] = \
                    (freq['frequency'] / freq['frequency'].sum()).cumsum()
                else:
                    freq['cumulative_percent'] = \
                    ((freq['frequency'] / freq['frequency'].sum()).cumsum()).round(decimals)

            if output != None:
                freq = freq.reset_index()
                freq = output.createDataFrame(freq)

        else: # Crosstab

            dataaa = data.copy()
            dataaa[row], dataaa[col] = dataaa[row].fillna(np.e), dataaa[col].fillna(np.e)
            freq = pd.pivot_table(dataaa, index=[row], columns=[col], aggfunc='size',
            fill_value=0).rename(columns={np.e: np.nan}, index={np.e: np.nan})

            if option == 'pct_col':
                for column in freq.columns:
                    if decimals == None: freq[column] = freq[column] / freq[column].sum()
                    else: freq[column] = (freq[column] / freq[column].sum()).round(decimals)

            if option == 'pct_row':
                suma = freq.sum(axis=1)
                for column in freq.columns:
                    if decimals == None: freq[column] = freq[column] / suma
                    else: freq[column] = (freq[column] / suma).round(decimals)

            if sort_col == '' or sort_col == row:
                if sort_dir == 'desc': freq = freq.sort_index(ascending=False)
            else:
                if sort_dir == 'desc': freq = freq.sort_values(sort_col, ascending=False)
                else: freq = freq.sort_values(sort_col)

            if output != None:
                freq.columns.names = [None]
                freq = freq.reset_index()
                freq = output.createDataFrame(freq)
                freq = freq.withColumnRenamed(row, row + '_' + col)

    else: # pyspark.DataFrame
        
        import pyspark.sql.functions as sf
        from pyspark.sql.types import IntegerType
        from pyspark.sql.types import FloatType
        from pyspark.sql.window import Window

        if col == '': # Frequency table

            freq = data.groupBy(row).count().withColumnRenamed('count', 'frequency')
            freq = freq.sort(row)

            if output != None:

                suma = freq.agg(sf.sum('frequency')).collect()[0][0]
                if decimals == None:
                    freq = freq.withColumn('percent',
                    sf.col('frequency') / sf.lit(suma))
                else:
                    freq = freq.withColumn('percent',
                    sf.format_number(sf.col('frequency') / sf.lit(suma), decimals))

                if sort_col == '':
                    if sort_dir == 'desc': freq = freq.sort(row, ascending=False)
                else:
                    if sort_dir == 'desc': freq = freq.sort(sort_col, ascending=False)
                    else: freq = freq.sort(sort_col)

                if cumulative == True:
                    freq = freq.withColumn('cumulative_frequency',
                    sf.sum('frequency').over(Window.rowsBetween(Window.unboundedPreceding, 0)))
                    if decimals == None: freq = freq.withColumn('cumulative_percent',
                    sf.sum(sf.col('frequency') / sf.lit(suma))\
                    .over(Window.rowsBetween(Window.unboundedPreceding, 0)))
                    else: freq = freq.withColumn('cumulative_percent',
                    sf.format_number(sf.sum(sf.col('frequency') / sf.lit(suma))\
                    .over(Window.rowsBetween(Window.unboundedPreceding, 0)), decimals))

            else:

                freq = freq.toPandas().set_index(row)

                if decimals == None: freq['percent'] = freq['frequency'] / freq['frequency'].sum()
                else: 
                    freq['percent'] = (freq['frequency'] / freq['frequency'].sum()).round(decimals)

                if sort_col == '' or sort_col == row:
                    if sort_dir == 'desc': freq = freq.sort_index(ascending=False)
                else:
                    if sort_dir == 'desc': freq = freq.sort_values(sort_col, ascending=False)
                    else: freq = freq.sort_values(sort_col)

                if cumulative == True:
                    freq['cumulative_frequency'] = freq['frequency'].cumsum()
                    if decimals == None:
                        freq['cumulative_percent'] = \
                        (freq['frequency'] / freq['frequency'].sum()).cumsum()
                    else:
                        freq['cumulative_percent'] = \
                        ((freq['frequency'] / freq['frequency'].sum()).cumsum()).round(decimals)

        else: # Crosstab

            freq = data.crosstab(row, col)

            if data.select(row).dtypes[0][1] in ('smallint', 'int', 'bigint'):
                freq = freq.withColumn(row + '' + col, sf.col(row + '' + col).cast(IntegerType()))
            elif data.select(row).dtypes[0][1] == 'double':
                freq = freq.withColumn(row + '' + col, sf.col(row + '' + col).cast(FloatType()))

            if data.select(col).dtypes[0][1] in ('smallint', 'int', 'bigint'):
                L1, L2 = [], []
                for i in freq.columns[1:]:
                    try: L1.append(int(i))
                    except: L2.append(i)
                L1.sort()
                L3 = L2 + [str(i) for i in L1]
                freq = freq.select([freq.columns[0]] + L3)
            elif data.select(col).dtypes[0][1] == 'double':
                L1, L2 = [], []
                for i in freq.columns[1:]:
                    try: L1.append(float(i))
                    except: L2.append(i)
                L1.sort()
                L3 = L2 + [str(i) for i in L1]
                freq = freq.select([freq.columns[0]] + L3)

            freq = freq.sort(row + '_' + col)

            if output != None:

                if option == 'pct_col':
                    for column in list(freq.columns[1:]):
                        if decimals == None: freq = freq.withColumn(
                        column, sf.col(column) / sf.sum(column).over(Window.partitionBy()))
                        else: freq = freq.withColumn(
                        column, sf.format_number(sf.col(column) / sf.sum(column)\
                        .over(Window.partitionBy()), decimals))

                if option == 'pct_row':
                    for column in list(freq.columns[1:]):
                        if decimals == None:
                            freq = freq.withColumn(column,
                            sf.col(column) / sum([sf.col(c) for c in freq.columns[1:]]))
                        else:
                            freq = freq.withColumn(column,
                            sf.format_number(sf.col(column) / sum([sf.col(c)
                            for c in freq.columns[1:]]), decimals))

                if sort_col == '':
                    if sort_dir == 'desc': freq = freq.sort(row + '_' + col, ascending=False)
                else:
                    if sort_dir == 'desc': freq = freq.sort(sort_col, ascending=False)
                    else: freq = freq.sort(sort_col)

            else:

                freq = freq.toPandas()
                freq = freq.rename(columns={row + '_' + col: row})
                freq = freq.set_index(row)
                freq.columns.name = col

                if option == 'pct_col':
                    for column in freq.columns:
                        if decimals == None: freq[column] = freq[column] / freq[column].sum()
                        else: freq[column] = (freq[column] / freq[column].sum()).round(decimals)

                if option == 'pct_row':
                    denominador = freq.sum(axis=1)
                    for column in freq.columns:
                        if decimals == None: freq[column] = freq[column] / denominador
                        else: freq[column] = (freq[column] / denominador).round(decimals)

                if sort_col == '' or sort_col == row:
                    if sort_dir == 'desc': freq = freq.sort_index(ascending=False)
                else:
                    if sort_dir == 'desc': freq = freq.sort_values(sort_col, ascending=False)
                    else: freq = freq.sort_values(sort_col)

    if type(freq) == type(pd.DataFrame([])) and len(values) > 0:

        for value in values:
            if value not in freq.index:
                freq.loc[value] = [0]*len(freq.columns)

        if sort_col == '' or sort_col == row:
            if sort_dir == 'desc': freq = freq.sort_index(ascending=False)
            else: freq = freq.sort_index() # Necesita reordenar sí o sí
        else:
            if sort_dir == 'desc': freq = freq.sort_values(sort_col, ascending=False)
            else: freq = freq.sort_values(sort_col)

    return freq

