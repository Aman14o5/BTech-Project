# Human rating of explanation quality

Each of the three team members rates the same 200 AI-written explanations, in a different order.
This gives the "human rating" metric of the research plan (Table 2), which RQ2 and RQ3 need.

## Which file is yours
| Rater | File |
|---|---|
| Rater 1 | `rating_sheet_rater1.xlsx` |
| Rater 2 | `rating_sheet_rater2.xlsx` |
| Rater 3 | `rating_sheet_rater3.xlsx` |

Agree among yourselves who is rater 1, 2 and 3. Don't put your names in the files, because this repo is public.

**Do not open `rating_key.csv`.** It says which model wrote each explanation and whether it failed our tests,
and knowing that would bias your ratings.

## How to fill it in
1. Open your file in Excel, or in Google Sheets (Drive → New → File upload, then open it with Google Sheets).
   The dropdowns work in both.
2. Read the **Instructions** tab once (about 3 minutes).
3. In the **Ratings** tab, for every row:
   - **column G (required):** Yes / Weakly Yes / Weakly No / No
   - **columns H–L (optional):** an `x` for each shortcoming that applies
   - **Notes (optional):** anything odd
4. Expect about 20–30 seconds per row, so roughly 1.5 hours in total. You can stop and continue later.
   Work alone, and don't discuss items until all three of you are done.
5. Don't sort, delete or add rows, and don't edit the Item column.

## Handing it back
Save or download the file as **.xlsx** with its original name. In Google Sheets: File → Download →
Microsoft Excel (.xlsx). Send it to Amey.

The sheets were built by `faithfulness/build_rating_sheets.py`. It uses 50 explanations per model, taken from
the 2,000 pairs used for the explanation-regeneration run, with a mix of labels and of explanations that pass
and fail the automatic tests.
