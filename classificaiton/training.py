
from pyspark.sql import SparkSession
spark = SparkSession.builder\
        .appName("classification_first")\
        .getOrCreate()

from pyspark.ml.classification import (RandomForestClassifier,
                                       GBTClassifier)

from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)

from pyspark.ml.tuning import (CrossValidator,
                               ParamGridBuilder)
    

def trainig_inference(
    classifier_name,
    df_train,
    df_test, features_list
  ):
    classifier = classifier_name

    binary_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol='prediction'
    )
    mc_evaluator = MulticlassClassificationEvaluator(
        metricName='accuracy'
    )

    param_grid = (ParamGridBuilder()\
                 .addGrid(
                     classifier.numTrees, [5, 10, 20]
                 )
                 .addGrid(
                     classifier.maxDepth, [2, 5, 10]
                 )\
                .build())
    cross_val = CrossValidator(
        estimator=classifier,
        estimatorParamMaps=param_grid,
        evaluator=MulticlassClassificationEvaluator(),
        numFolds=3
    )
    fit_model = cross_val.fit(df_train)
    best_model = fit_model.bestModel

    predictions = fit_model.transform(df_test)
    acc = mc_evaluator.evaluate(predictions)
    print(f'accuracy of the model on test data is {acc}\n')


    feature_imp = [float(i) for i in best_model.featureImportances.toArray()]
    fi_df = spark.createDataFrame(zip(
        features_list,
        feature_imp), schema=['features', 'feature_importances']
    )

    fi_df = fi_df.orderBy(fi_df['feature_importances'].desc())
    return best_model, fi_df, predictions


    

    
