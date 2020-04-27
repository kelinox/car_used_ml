using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace carUserPredictionPrice
{
    class Program
    {

        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "assets", "training_data.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "assets", "test_data.csv");

        static void Main(string[] args)
        {
            MLContext context = new MLContext(seed: 0);

            var model = Train(context, _trainDataPath);

            Evaluate(context, model);

            TestSinglePrediction(context, model);
        }

        private static ITransformer Train(MLContext context, string trainDataPath)
        {
            IDataView dataView = context.Data.LoadFromTextFile<UsedCar>(trainDataPath, hasHeader: true, separatorChar: ',');
            IDataView dataViewShuffled = context.Data.ShuffleRows(dataView);

            var pipeline = context.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "SalePrice")
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "CarManufacturerEncoded", inputColumnName: "CarManufacturer"))
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "CarModelEncoded", inputColumnName: "CarModel"))
                .Append(context.Transforms.Concatenate(
                    outputColumnName: "Features",
                    "CarManufacturerEncoded",
                    "CarModelEncoded",
                    "ReleaseYear",
                    "Kilometers"
                ))
                .Append(context.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataViewShuffled);

            return model;
        }

        private static void Evaluate(MLContext context, ITransformer model)
        {
            IDataView dataView = context.Data.LoadFromTextFile<UsedCar>(_testDataPath, hasHeader: true, separatorChar: ',');

            var predictions = model.Transform(dataView);

            var metrics = context.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        private static void TestSinglePrediction(MLContext context, ITransformer model)
        {
            var predictionFunction = context.Model.CreatePredictionEngine<UsedCar, UserCarPrediction>(model);

            var usedCar = new UsedCar
            {
                CarManufacturer = "Audi",
                CarModel = "A3",
                Kilometers = 85300f,
                ReleaseYear = 2014f,
                SalePrice = 0f
            };

            var prediction = predictionFunction.Predict(usedCar);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted sale price: {prediction.SalePrice:0.####}, actual sale price: 14 490");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
