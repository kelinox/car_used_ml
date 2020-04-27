using Microsoft.ML.Data;

namespace carUserPredictionPrice
{
    class UsedCar
    {
        [LoadColumn(0)]
        public string CarManufacturer { get; set; }

        [LoadColumn(1)]
        public string CarModel { get; set; }

        [LoadColumn(2)]
        public float ReleaseYear { get; set; }

        [LoadColumn(3)]
        public float Kilometers { get; set; }

        [LoadColumn(4)]
        public float SalePrice { get; set; }
    }

    class UserCarPrediction
    {
        [ColumnName("Score")]
        public float SalePrice;
    }
}