import numpy as np, pandas as pd

from IPython.display import display

####################################################################################################


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
