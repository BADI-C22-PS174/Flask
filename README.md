# Documentation API

  ```sh
{
	"info": {
		"_postman_id": "3349b46b-b0f8-4005-be3b-420e329775f4",
		"name": "BADI-API",
		"schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
	},
	"item": [
		{
			"name": "Adding API",
			"request": {
				"method": "GET",
				"header": [],
				"url": {
					"raw": "http://34.101.162.19/",
					"protocol": "http",
					"host": [
						"34",
						"101",
						"162",
						"19"
					],
					"path": [
						"db"
					]
				},
				"description": "[GET] connect to API"
			},
			"response": []
		},
		{
			"name": "Get Data",
			"request": {
				"method": "GET",
				"header": [],
				"url": {
					"raw": "http://34.101.162.19/db",
					"protocol": "http",
					"host": [
						"34",
						"101",
						"162",
						"19"
					],
					"path": [
						"db"
					]
				}
			},
			"response": []
		},
		{
			"name": "Get Prediction From Data",
			"request": {
				"method": "GET",
				"header": [],
				"url": {
					"raw": "http://34.101.162.19/prediction/sales_forcasting.csv",
					"protocol": "http",
					"host": [
						"34",
						"101",
						"162",
						"19"
					],
					"path": [
						"prediction",
						"sales_forcasting.csv"
					]
				}
			},
			"response": []
		}
	]
}
```
