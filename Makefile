.PHONY: run

run:
	git config pull.rebase true
	mkdir -p data/zip
	wget -P data/zip https://lodmedia.hb.bizmrg.com/case_files/1128568/train_dataset_soc_dem_train.zip
	wget -P data/zip https://lodmedia.hb.bizmrg.com/case_files/1128568/test_dataset_Тест.zip
	unzip data/zip/train_dataset_soc_dem_train.zip -d data 
	unzip data/zip/test_dataset_Тест.zip -d data
	mv data/Описание\ данных.pdf data/data_desc.pdf
	rm data/baseline_socdem.ipynb


