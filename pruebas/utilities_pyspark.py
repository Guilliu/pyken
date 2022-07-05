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


def compute_ks_pyspark(df, target, score):
    
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

