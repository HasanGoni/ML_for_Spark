
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName('RegressionExercise')\
    .getOrCreate()


def change_label_name(
        df,
        present_name,
        new_name='label'):
    """Changing dependent variable name as new_name

    Args:
        df (DataFrame): This dataframes name will be 
                        changed
        present_name (string): a string which is old name
                               of the dependent variable column
        new_name (string): a string which will be new
                           new_name,default value is 'label'
    """

    df = df.withColumnRenamed(
        present_name,
        new_name
    )
    return df


def change_label_type(
        df,
        dependent_column='label'):
    """converting dependent varaible type to float

    Args:
        df (DataFrame): DataFrame which dependent 
                        varialbles types need to change
        dependent_column (str): Name of the column. Defaults to 'label'.

    Returns:
        DataFrame: Return the Dataframe which 
                   has a datatype Float of the label
    """

    if str(df.schema[dependent_column].dataType) != 'IntegerType':
        df = df.withColumn(
            dependent_column,
            df[dependent_column].cast(FloatType())
        )
    return df


def cont_cat_split(
        df, input_columns):

    cat_columns = []
    continuous_columns = []

    for col in input_columns:
        if str(df.schema[col].dataType) == 'StringType':
            new_name = f'{col}_num'
            cat_columns.append(new_name)
            indexer = StringIndexer(
                inputCol=col,
                outputCol=new_name)
            df_new = indexer.fit(df).transform(df)
        else:
            continuous_columns.append(col)
            df_new = df
    return df_new, cat_columns, continuous_columns


def create_skew_dict(
    df_new,
    cont_columns
):
    """Create approxQuantile from numeric dataframe

    Args:
        df_new (DataFrame): DataFrame to detect skewness
        cont_columns (continuous columns): conitnuous columns

    Returns:
        Dictionary: dictionary
    """

    d = {}
    for i in cont_columns:
        d[i] = df_new.approxQuantile(
            i, [0.01, 0.99], 0.25
        )
    return d


def treat_outlier(df_new, cont_columns, dict_):

    for col in cont_columns:
        skewness_ = df_new.agg(
            skewness(df_new[col])
        ).collect()
        skew = skewness_[0][0]

        if skew > 1:

            df_new = df_new.withColumn(
                col,
                log(when(df_new[col] < dict_[col][0], dict_[col][0])
                    .when(df_new[col] > dict_[col][1], dict_[col][1])
                    .otherwise(df_new[col]) + 1).alias(col)
            )

            print(f'=={col}== has right skewness, skew = {skew}')

        elif skew < -1:

            df_new = df_new.withColumn(
                col,
                exp(when(df[col] < dict_[col][0], dict_[col][0])
                    .when(df_new[col] < dict_[col][1], dict_[col][1])
                    .otherwise(df_new[col])).alias(col)
            )
            print(f'=={col}== has left skewness, skew = {skew}')

    return df_new


def vectorization_data(
        df_new,
        cont_columns,
        cat_columns):

    all_columns = cont_columns + cat_columns

    assembler = VectorAssembler(
        inputCols=all_columns,
        outputCol='features')

    df_new = assembler.transform(df_new)\
        .select('features', 'label')

    return df_new


def training_and_inference(df_new,
                           all_columns,
                           train_split=0.7, test_split=0.3,
                           model_name='random_forest'):

    df_train, df_test = df_new.randomSplit([train_split, test_split])

    print(f' train and test split size')
    print(df_train.count(), df_test.count())
    if model_name == 'random_forest':
        rf = RandomForestRegressor(
        )
    elif model_name == 'gbt_regressor':
        rf = GBTRegressor()

    param_grid = (ParamGridBuilder()
                  .addGrid(rf.numTrees, [10, 20, 30])
                  .addGrid(rf.maxDepth, [2, 5, 10, 20])
                  .build())
    evaluator = RegressionEvaluator(metricName='rmse')
    cross_val = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3
    )

    fitModel = cross_val.fit(df_train)
    best_model = fitModel.bestModel

    predictions_ = fitModel.transform(df_test)

    rmse_ = evaluator.evaluate(predictions_)

    print(f' rmse for this model on test data = {rmse_}\n')

    rf_fi = best_model.featureImportances.toArray()
    rf_fi_ = [float(i) for i in rf_fi]

    rf_fi_df = spark.createDataFrame(zip(
        all_columns, rf_fi_),
        schema=['features', 'values'])

    return best_model, rf_fi_df


def preprocessing_csv(
    csv_file_name: str,
    dependent_column_name: str,
    treat_outliers=False,
    train_split=0.7,
    test_split=0.3,
    model_name='random_forest'  # gbt_regressor can also be written
):

    df = spark.read.csv(
        csv_file_name,
        inferSchema=True,
        header=True)

    # Printing data frame
    print(df.limit(5).toPandas())

    input_columns = df.columns
    input_columns.remove(dependent_column_name)

    print(f'input columns are == {input}')

    print(f'total dataframe rows = {df.count()}')
    df = df.na.drop()
    print(f' dataframe rows after na removal = {df.count()}')

    print('changing dependent variable name to label')
    df = change_label_name(
        df,
        dependent_column_name,
    )
    print('changing dependent variable label type')

    df = change_label_type(df)

    print(f' converting columns to have continous and string varialbles\n')
    df_new, cat_columns, cont_columns = cont_cat_split(df, input_columns)
    print(f'continous variables are {cont_columns}\n')
    print(f'string variables are {cat_columns}\n')

    dic_ = create_skew_dict(df_new, cont_columns)

    if treat_outliers:
        df_new = treat_outlier(
            df_new,
            cont_columns,
            dic_)

    print(f'vectorizing started == \n')
    df_new = vectorization_data(
        df_new,
        cont_columns,
        cat_columns)

    print(df_new.show(5))
    print(f'\n splitting data to train and test split')

    best_model, rf_fi = training_and_inference(df_new,
                                               all_columns=input_columns,
                                               model_name=model_name)
    print(rf_fi.orderBy(rf_fi['values'].desc()).show())

    return best_model, rf_fi
