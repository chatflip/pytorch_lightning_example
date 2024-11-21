.PHONY: mypy

mypy:
	mypy . --namespace-packages --explicit-package-bases \
		--exclude "1_classification_food101/src/__init__.py" \
		--exclude "1_classification_food101/src/train.py" \
		--exclude "1_classification_food101/src/utils.py" \
		--exclude "5_segmentation_voc/src/__init__.py" \
		--exclude "5_segmentation_voc/src/train.py" \
		--exclude "5_segmentation_voc/src/utils.py" 
