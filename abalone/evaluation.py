{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "375d03db-6e73-4b57-9ef9-542cf5cd0588",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pathlib\n",
    "import pickle\n",
    "import tarfile\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import xgboost\n",
    "\n",
    "# This script is used in a processing step to perform model evaluation. \n",
    "# It takes a trained model and the test dataset as input, then produces a JSON \n",
    "# file containing classification evaluation metrics.\n",
    "\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    model_path = f\"/opt/ml/processing/model/model.tar.gz\"\n",
    "    with tarfile.open(model_path) as tar:\n",
    "        tar.extractall(path=\".\")\n",
    "    \n",
    "    model = pickle.load(open(\"xgboost-model\", \"rb\"))\n",
    "\n",
    "    test_path = \"/opt/ml/processing/test/test.csv\"\n",
    "    df = pd.read_csv(test_path, header=None)\n",
    "    \n",
    "    y_test = df.iloc[:, 0].to_numpy()\n",
    "    df.drop(df.columns[0], axis=1, inplace=True)\n",
    "    \n",
    "    X_test = xgboost.DMatrix(df.values)\n",
    "    \n",
    "    predictions = model.predict(X_test)\n",
    "\n",
    "    mse = mean_squared_error(y_test, predictions)\n",
    "    std = np.std(y_test - predictions)\n",
    "    report_dict = {\n",
    "        \"regression_metrics\": {\n",
    "            \"mse\": {\n",
    "                \"value\": mse,\n",
    "                \"standard_deviation\": std\n",
    "            },\n",
    "        },\n",
    "    }\n",
    "\n",
    "    output_dir = \"/opt/ml/processing/evaluation\"\n",
    "    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)\n",
    "    \n",
    "    evaluation_path = f\"{output_dir}/evaluation.json\"\n",
    "    with open(evaluation_path, \"w\") as f:\n",
    "        f.write(json.dumps(report_dict))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "conda_python3",
   "language": "python",
   "name": "conda_python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
