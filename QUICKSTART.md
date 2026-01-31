# Quick Start Guide - Complete Your CAFA-6 Project

## Step 1: Run Notebook Evaluation Cells

Open `02_notebook/cafa6_baseline.ipynb` in VS Code and execute these two cells:

### Cell 45: Validation Evaluation
- Location: The cell starting with `# Validation split evaluation using weighted Fmax`
- What it does: Computes Fmax, Precision, Recall metrics on held-out validation set
- Expected output: `Validation Fmax: X.XXXX, Precision: X.XXXX, Recall: X.XXXX`
- **Action:** Copy these numbers to Section 5.3 in the report

### Cell 46: Export Artifacts
- Location: The cell starting with `# Persist artifacts for the app`
- What it does: Saves vectorizer.pkl, X_train.npz, train_ids.json, train_term_map.json
- Expected output: Confirmation messages showing file paths
- **Result:** Check `05_model/artifacts/` has 4 files

**How to run:**
1. Open the notebook
2. Select each cell (click on it)
3. Press `Shift+Enter` or click the play button
4. Wait for output (Cell 45 may take 2-3 minutes)

---

## Step 2: Test the Application

After artifacts are generated, verify the app works:

```bash
cd E:\Computational Intelligence\CIS6005_Kaggle_Project_Folder_Structure
venv\Scripts\activate
streamlit run 03_app/app.py
```

1. Paste a test sequence or upload a FASTA
2. Verify predictions appear
3. Download the TSV file

---

## Step 3: Take Screenshots

Capture these screenshots and save to `01_screenshots/`:

### Competition Proof
- [ ] `competition_page.png` - CAFA-6 page showing 2025 dates
- [ ] `enrollment_proof.png` - Your team/enrollment confirmation
- [ ] `rules_acknowledgement.png` - Competition rules accepted

### Submissions
- [ ] `public_submission.png` - Public leaderboard with your score
- [ ] `private_submission.png` - Final/private leaderboard (after deadline)

### App Demo
- [ ] `app_home.png` - Streamlit app with input area
- [ ] `app_results.png` - Prediction table with GO terms
- [ ] `app_download.png` - Download TSV button

**Screenshot shortcuts:**
- Windows: `Win+Shift+S` (Snip & Sketch)
- Full screen: `PrtScn` key

---

## Step 4: Update Report with Results

Edit `04_report/REPORT_DRAFT.md`:

1. Find Section 5.3 "Results"
2. Replace placeholders with actual metrics from Cell 45:
   ```markdown
   **Validation Performance:**
   - **F-max:** 0.XXXX  ← Fill from notebook output
   - **Precision:** 0.XXXX
   - **Recall:** 0.XXXX
   ```

3. Add your Student ID at the top:
   ```markdown
   **Student ID:** st12345678  ← Replace with yours
   ```

---

## Step 5: Convert Report to PDF

### Option A: Using Microsoft Word
1. Open `04_report/REPORT_DRAFT.md` in VS Code
2. Copy all content
3. Paste into Word
4. Format headings (Heading 1, 2, 3)
5. Insert screenshots in Appendix D
6. Save As PDF → `04_report/CIS6005_Report.pdf`

### Option B: Using Pandoc (if installed)
```bash
cd 04_report
pandoc REPORT_DRAFT.md -o CIS6005_Report.pdf --pdf-engine=xelatex
```

### Option C: Using Google Docs
1. Upload REPORT_DRAFT.md to Google Drive
2. Open with Google Docs
3. Format and insert screenshots
4. File → Download → PDF

---

## Step 6: Final Checklist

Before submission, verify:

- [ ] 05_model/artifacts/ has 4 files (vectorizer.pkl, X_train.npz, etc.)
- [ ] 05_model/submission.tsv exists and has no header
- [ ] 01_screenshots/ has all required images
- [ ] 04_report/CIS6005_Report.pdf is <4000 words
- [ ] Report Section 5.3 has actual Fmax/Precision/Recall numbers
- [ ] Student ID is filled in report header
- [ ] Screenshots are embedded or referenced in Appendix D
- [ ] References are properly formatted

---

## Troubleshooting

**Notebook cell fails with import error:**
```bash
pip install joblib scipy scikit-learn tqdm
```

**App won't start:**
- Check artifacts exist in `05_model/artifacts/`
- Rerun Cell 46 in notebook
- Verify `streamlit` is installed: `pip list | findstr streamlit`

**"No module named 'src'":**
- The notebook adds `05_model` to sys.path
- Ensure you're running from `02_notebook/` directory

**Validation cell takes too long:**
- Expected: 2-3 minutes for ~100k proteins
- If >10 minutes, reduce K to 3 or max_features to 50000

---

## Quick Command Reference

```bash
# Activate environment
venv\Scripts\activate

# Install missing packages
pip install -r 05_model/requirements.txt

# Run app
streamlit run 03_app/app.py

# Check artifacts
dir 05_model\artifacts

# Verify submission format
python -c "import pandas as pd; df=pd.read_csv('05_model/submission.tsv', sep='\t', header=None); print(f'Shape: {df.shape}, Columns: {df.shape[1]}, Score range: [{df[2].min()}, {df[2].max()}]')"
```

---

**Estimated Time:**
- Run notebook cells: 5 minutes
- Take screenshots: 10 minutes
- Update report: 5 minutes
- Convert to PDF: 10 minutes
- **Total: ~30 minutes**

Good luck with your submission! 🚀
