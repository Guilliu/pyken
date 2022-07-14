import pandas as pd, statsmodels, datetime

from .todd import *
from .diane import *

####################################################################################################


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
method='stepwise', metric='pvalue', threshold=0.01, criterio_stop_ks_gini=True, 
max_iters=12, included_vars=[], muestra_test=None, show='gini', log_mode=False):
    
    N = 150
    
    if log_mode: file_prints = open('log_modelo.txt', 'a')
    else: file_prints = None

    if features != []: included_vars, max_iters = features, 0

    if method not in ('forward', 'stepwise'):
        raise ValueError("Valor inválido para el parámetro 'method'. "
        "Solo están pertimidos los valores 'forward' y 'stepwise'")

    if metric not in ('pvalue', 'ks', 'gini'):
        raise ValueError("Valor inválido para el parámetro 'metric'. "
        "Solo están pertimidos los valores 'pvalue', 'ks' y 'gini")

    if max_iters > len(var_list):
        print('Cuidado, has puesto un valor numero máximo de iteraciones ({})'
        ' superior al número de variables candidatas ({})'.format(max_iters, len(var_list)), file=file_prints)
        print('-' * N, file=file_prints)
        max_iters = len(var_list)

    old_ks, old_gini = 0, 0
    
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
                        print('Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'KS train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1], ks_train*100, var), file=file_prints)
                    if show == 'gini':
                        print('Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1], gini_train*100, var), file=file_prints)
                    if show == 'both':
                        print('Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'KS train = {:.2f}% | Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1],
                        ks_train*100, gini_train*100, new_var), file=file_prints)

                else:
                    if show == 'ks':
                        print('Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1], ks_train*100, ks_test*100, new_var), file=file_prints)
                    if show == 'gini':
                        print('Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'Gini train = {:.2f}% | Gini test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalues[-1],
                        gini_train*100, gini_test*100, new_var), file=file_prints)
                    if show == 'both':
                        print('Step {} | Time - 0:00:00.000000 | p-value = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% | Gini train = {:.2f}% | Gini test '
                        '= {:.2f}% ---> Feature selected: {}'.format(str(i+1).zfill(2), pvalues[-1],
                        ks_train*100, ks_test*100, gini_train*100, gini_test*100, new_var), file=file_prints)


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
                        print('Step {} | Time - 0:00:00.000000 | KS train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), ks_train*100, new_var), file=file_prints)
                    if show == 'gini':
                        print('Step {} | Time - 0:00:00.000000 | Gini train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), gini_train*100, new_var), file=file_prints)
                    if show == 'both':
                        print('Step {} | Time - 0:00:00.000000 | KS train = {:.2f}% | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), ks_train*100, gini_train*100, new_var), file=file_prints)
                else:
                    if show == 'ks':
                        print('Step {} | Time - 0:00:00.000000 | KS train = {:.2f}% | '
                        'KS test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), ks_train*100, ks_test*100, new_var), file=file_prints)
                    if show == 'gini':
                        print('Step {} | Time - 0:00:00.000000 | Gini train = {:.2f}% | '
                        'Gini test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), gini_train*100, gini_test*100, new_var), file=file_prints)
                    if show == 'both':
                        print('Step {} | Time - 0:00:00.000000 | KS train = {:.2f}% | '
                        'KS test = {:.2f}% | Gini train = {:.2f}% | Gini test = {:.2f}% '
                        '---> Feature selected: {}'.format(str(i+1).zfill(2), ks_train*100,
                        ks_test*100, gini_train*100, gini_test*100, new_var), file=file_prints)

        else:

            if method == 'forward':

                if metric not in ('ks', 'gini'):
                    raise ValueError("El método 'forward' "
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
                
                if metric == 'ks':
                    if ks_train <= old_ks+0.0020:
                        print('-' * N, file=file_prints)
                        print('En el siguiente paso el KS no sube '
                        'ya más de un 0.20, detenemos el proceso', file=file_prints)
                        features.pop()
                        break
                
                elif metric == 'gini':
                    if gini_train <= old_gini+0.0030:
                        print('-' * N, file=file_prints)
                        print('En el siguiente paso el Gini no sube '
                        'ya más de un 0.30, detenemos el proceso', file=file_prints)
                        features.pop()
                        break
                    
                old_ks, old_gini = ks_train, gini_train
                
                if not isinstance(muestra_test, type(None)):
                    test_final, ks_test, gini_test = apply_scorecard(
                    muestra_test, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)

                if isinstance(muestra_test, type(None)):
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print('Step {} | Time - {} | KS train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, ks_train*100, new_var), file=file_prints)
                    if show == 'gini':
                        print('Step {} | Time - {} | Gini train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, gini_train*100, new_var), file=file_prints)
                    if show == 'both':
                        print('Step {} | Time - {} | KS train = {:.2f}% | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, ks_train*100, gini_train*100, new_var), file=file_prints)
                else:
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print('Step {} | Time - {} | KS train = {:.2f}% | '
                        'KS test = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, ks_train*100, ks_test*100, new_var), file=file_prints)
                    if show == 'gini':
                        print('Step {} | Time - {} | Gini train = {:.2f}% | '
                        'Gini test = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, gini_train*100, gini_test*100, new_var), file=file_prints)
                    if show == 'both':
                        print('Step {} | Time - {} | KS train = {:.2f}% | '
                        'KS test = {:.2f}% | Gini train = {:.2f}% | Gini test = {:.2f}% '
                        '---> Feature selected: {}'.format(str(i+1).zfill(2), end - start,
                        ks_train*100, ks_test*100, gini_train*100, gini_test*100, new_var), file=file_prints)

            elif method == 'stepwise':

                if metric != 'pvalue':
                    raise ValueError("El método 'stepwise' "
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
                    print('-' * N, file=file_prints)
                    print('Ya ninguna variable tiene un p-valor'
                    ' < {}, detenemos el proceso.'.format(threshold), file=file_prints)
                    break

                new_var = aux.iloc[0]['var']
                features.append(new_var)

                scorecard, features_length, pvalues = compute_scorecard(
                data, features, info, target_name=target_name, pvalues=True)
                new_pvalue = pvalues[-1]
                train_final, ks_train, gini_train = apply_scorecard(
                data, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)
                
                if criterio_stop_ks_gini:
                    if ks_train <= old_ks+0.0020 and gini_train <= old_gini+0.0030:
                        print('-' * N, file=file_prints)
                        print('En el siguiente paso ni el KS ni el GINI del train '
                        'suben ya más de un 0.20 o un 0.30 respectivamente, detenemos el proceso', file=file_prints)
                        features.pop()
                        break
                old_ks, old_gini = ks_train, gini_train
                
                if not isinstance(muestra_test, type(None)):
                    test_final, ks_test, gini_test = apply_scorecard(
                    muestra_test, scorecard, info, metrics=['ks', 'gini'], target_name=target_name)

                if isinstance(muestra_test, type(None)):
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print('Step {} | Time - {} | p-value = {:.2e} | '
                        'KS train = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, new_pvalue, ks_train*100, new_var), file=file_prints)
                    if show == 'gini':
                        print('Step {} | Time - {} | p-value = {:.2e} | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, new_pvalue, gini_train*100, new_var), file=file_prints)
                    if show == 'both':
                        print('Step {} | Time - {} | p-value = {:.2e} | '
                        'KS train = {:.2f}% | Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), end - start, new_pvalue,
                        ks_train*100, gini_train*100, new_var), file=file_prints)

                else:
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print('Step {} | Time - {} | p-value = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), end - start, new_pvalue,
                        ks_train*100, ks_test*100, new_var), file=file_prints)
                    if show == 'gini':
                        print('Step {} | Time - {} | p-value = {:.2e} | '
                        'Gini train = {:.2f}% | Gini test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), end - start, new_pvalue,
                        gini_train*100, gini_test*100, new_var), file=file_prints)
                    if show == 'both':
                        print('Step {} | Time - {} | p-value = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% | Gini train = {:.2f}% | Gini test '
                        '= {:.2f}% ---> Feature selected: {}'.format(str(i+1).zfill(2),
                        end - start, new_pvalue, ks_train*100, ks_test*100,
                        gini_train*100, gini_test*100, new_var), file=file_prints)

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
                        old_ks, old_gini = ks_train, gini_train
                        if not isinstance(muestra_test, type(None)):
                            test_final, ks_test, gini_test = apply_scorecard(
                            muestra_test, scorecard, info,
                            metrics=['ks', 'gini'], target_name=target_name)

                            if isinstance(muestra_test, type(None)):
                                end = datetime.datetime.now()
                                if show == 'ks':
                                    print('Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | KS train = {:.2f}% ---> Feature deleted : {}'\
                                    .format(str(i+1).zfill(2), dict_pvalues[v], ks_train*100, v), file=file_prints)
                                if show == 'gini':
                                    print('Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | Gini train = {:.2f}% ---> Feature deleted : {}'\
                                    .format(str(i+1).zfill(2), dict_pvalues[v], gini_train*100, v), file=file_prints)
                                if show == 'both':
                                    print('Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | KS train = {:.2f}% | Gini train '
                                    '= {:.2f}% ---> Feature deleted : {}'.format(str(i+1).zfill(2),
                                    dict_pvalues[v], ks_train*100, gini_train*100, v), file=file_prints)

                            else:
                                end = datetime.datetime.now()
                                if show == 'ks':
                                    print('Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | KS train = {:.2f}% | KS test = {:.2f}% '
                                    '---> Feature deleted : {}'.format(str(i+1).zfill(2),
                                    dict_pvalues[v], ks_train*100, ks_test*100, v), file=file_prints)
                                if show == 'gini':
                                    print('Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | Gini train = {:.2f}% | Gini test = {:.2f}% '
                                    '---> Feature deleted : {}'.format(str(i+1).zfill(2),
                                    dict_pvalues[v], gini_train*100, gini_test*100, v), file=file_prints)
                                if show == 'both':
                                    print('Step {} | Time - 0:00:00.000000 | p-value '
                                    '= {:.2e} | KS train = {:.2f}% | KS test = {:.2f}% | '
                                    'Gini train = {:.2f}% | Gini test = {:.2f}% ---> Feature '
                                    'deleted : {}'.format(str(i+1).zfill(2), dict_pvalues[v],
                                    ks_train*100, ks_test*100, gini_train*100, gini_test*100, v), file=file_prints)

    print('-' * N, file=file_prints)
    print('Selección terminada: {}'.format(features), file=file_prints)
    print('-' * N, file=file_prints)
    
    if log_mode: file_prints.close()

    return features


def display_table_ng(modelo_newgroups, candidate_var, objeto, bp):

    if not isinstance(bp, dict):
        vector = data_convert(modelo_newgroups.X_train[candidate_var].values, string_categories2(bp))[3]
        breakpoints_num = breakpoints_to_num(bp)
        groups_names = compute_group_names(objeto.dtype, bp)
        display(compute_table(vector, modelo_newgroups.y_train, breakpoints_num, groups_names))

    else:
        vector = remapeo_missing(data_convert(modelo_newgroups.X_train[candidate_var].values, string_categories2(bp))[3], bp)
        breakpoints_num = breakpoints_to_num(bp['breakpoints'])
        groups_names = compute_group_names(objeto.dtype, bp['breakpoints'], bp['missing_group'])                                 
        display(compute_table(vector, modelo_newgroups.y_train, breakpoints_num, groups_names))
        
