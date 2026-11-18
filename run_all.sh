set -e
pip install -r requirements.txt
python collect_moex.py
python collect_girbo.py
python build_dataset.py
python train.py
streamlit run app.py
