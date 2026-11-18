pip install -r requirements.txt
python collect_moex.py || exit /b 1
python collect_girbo.py || exit /b 1
python build_dataset.py || exit /b 1
python train.py || exit /b 1
streamlit run app.py
