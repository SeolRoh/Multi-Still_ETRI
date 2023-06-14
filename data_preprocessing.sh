echo "start KEMDy data preprocessing..."
python KEMDy_preprocessing.py # KEMDy19, KEMD20 데이터셋 읽고, 저장

echo "start Data Balancing"
python Data_Balancing.py # 음성데이터가 존재하지 않는 데이터 정리 및 train, test 분리
