using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;

namespace  Recommendation
{
    public class MovieRating
    {
        [LoadColumn(0) ]
        public float userId;

        [LoadColumn(1) ]
        public float movieId;
        
        [LoadColumn(2) ]
        public float Label;
        
    }

     public class MovieRatingPrediction
    {
        public float Label;

        public float Score;
    }

    public class Program
    {
        public static IDataView PreprocessData(MLContext mlContext, IDataView dataView)
        {
            return mlContext.Transforms.Conversion.MapValueToKey(outputColumnName:"userId",inputColumnName:"userId" )
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName:"movieId",inputColumnName:"movieId" ))
                .Fit(dataView).Transform(dataView);
        
        }
        public static void SaveData(MLContext mlContext, IDataView dataView, string dataPath)
        {
            using (var fileStream = new FileStream(dataPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Data.SaveAsText(dataView, fileStream, separatorChar:',', headerRow: true, schema: false);
            }
        }

    static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            var dataPath = "preprocessed_ratings.csv";
            IDataView fullData = mlContext.Data.LoadFromTextFile<MovieRating>(dataPath, hasHeader: true, separatorChar: ',');
            var trainTestData = mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2);
            IDataView trainData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;
            return (trainData, testData);



        }

public static void PrintDataPreview(IDataView dataView)
        {
            var preview = dataView.Preview();
            foreach(var row in preview.RowView)
            {
                foreach(var column in row.Values)
                {
                    Console.Write($"{column.Key}; {column.Value}");
                }
                Console.WriteLine();
            }
        }


        static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "outputUserId", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "outputMovieId", inputColumnName:"movieId"));

                var options = new MatrixFactorizationTrainer.Options
                {
                    MatrixColumnIndexColumnName = "outputUserId",
                    MatrixRowIndexColumnName = "outputMovieId",
                    LabelColumnName = "Label",
                    NumberOfIterations = 20,
                    ApproximationRank = 100,
                };
                var trainEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));
                ITransformer model = trainEstimator.Fit(trainingDataView);
                Console.WriteLine("Model successfully trained");

                return model;
        }
        
        static void EvaluateModel(MLContext mlContext, IDataView testDataView,ITransformer model)
        {
            var prediction = model.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());   
        }

        static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            var testInput = new MovieRating {userId =  14, movieId = 433};
            var movieRatingPrediction = predictionEngine.Predict(testInput);
            Console.WriteLine("Predicted rating for movie" + testInput.movieId + "is " + Math.Round(movieRatingPrediction.Score, 1));
            string recommendation = Math.Round(movieRatingPrediction.Score, 1) > 3.5 ?
                "Movie " + testInput.movieId + " is recommended for user " + testInput.userId : 
                "Movie " + testInput.movieId + " is not recommended for user " + testInput.userId;
            Console.WriteLine(recommendation);

        }



        
        public static void Main()
        {
            MLContext mlContext = new MLContext();
            IDataView fullData = mlContext.Data.LoadFromTextFile<MovieRating>("ratings.csv", hasHeader: true, separatorChar: ',');
            IDataView preprocessData = PreprocessData(mlContext, fullData);        
            SaveData(mlContext, preprocessData, "preprocessed_ratings.csv");
            (IDataView trainingDataView, IDataView testDataView) data = LoadData(mlContext);
            PrintDataPreview(data.trainingDataView);
            ITransformer model = TrainModel(mlContext, data.trainingDataView);
            EvaluateModel(mlContext, data.testDataView, model);
            UseModelForSinglePrediction(mlContext, model);



            
            
        }






    }



}
