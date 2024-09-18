import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

# Initialize FastAPI
app = FastAPI()

# Load the dataset once when the server starts
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()  # Return an empty DataFrame if the file is not found
    return df

data_path = r"C:\Users\user\Desktop\Project\Employees.csv"
df = load_data(data_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Employee Salary API"}

@app.get("/salary_stats")
def salary_statistics():
    if df.empty:
        raise HTTPException(status_code=500, detail="No data available")

    # Check if necessary columns exist
    if 'Gender' not in df.columns or 'Monthly Salary' not in df.columns:
        raise HTTPException(status_code=500, detail="Required columns are missing in the data")

    # Ensure 'Monthly Salary' is numeric and handle errors
    df['Monthly Salary'] = pd.to_numeric(df['Monthly Salary'], errors='coerce')

    # Calculate salaries by gender
    male_salaries = df[df['Gender'] == 'Male']['Monthly Salary'].dropna().values
    female_salaries = df[df['Gender'] == 'Female']['Monthly Salary'].dropna().values

    total_male_salaries = np.sum(male_salaries)
    total_female_salaries = np.sum(female_salaries)

    # Avoid division by zero in percentage calculation
    if total_female_salaries == 0:
        if total_male_salaries == 0:
            percentage_difference = 0  # Both salaries are zero
        else:
            percentage_difference = float('inf')  # Infinite difference if only males have salaries
    else:
        percentage_difference = ((total_male_salaries - total_female_salaries) / total_female_salaries) * 100

    # Determine which gender has a higher total salary
    higher_salary = "Males" if total_male_salaries > total_female_salaries else "Females" if total_female_salaries > total_male_salaries else "Equal"

    result = {
        "total_male_salaries": total_male_salaries,
        "total_female_salaries": total_female_salaries,
        "percentage_difference": percentage_difference,
        "higher_salary": higher_salary
    }

    return result

@app.get("/salary_distribution")
def salary_distribution():
    if df.empty:
        raise HTTPException(status_code=500, detail="No data available")

    plt.figure(figsize=(15, 10))

    # Distribution of Monthly Salary
    plt.subplot(2, 2, 1)
    sns.histplot(df['Monthly Salary'], kde=True)
    plt.title('Distribution of Monthly Salary')

    # Box plot of Monthly Salary by Department
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Department', y='Monthly Salary', data=df)
    plt.title('Monthly Salary by Department')
    plt.xticks(rotation=45)

    # Correlation heatmap for Years, Monthly Salary, and Overtime Hours
    plt.subplot(2, 2, 3)
    if 'Years' in df.columns and 'Overtime Hours' in df.columns:
        corr = df[['Years', 'Monthly Salary', 'Overtime Hours']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    else:
        plt.text(0.5, 0.5, 'Not enough data for correlation heatmap', horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.title('Correlation Heatmap')

    # Bar plot of Average Monthly Salary by Gender
    plt.subplot(2, 2, 4)
    sns.barplot(x='Gender', y='Monthly Salary', data=df, estimator=np.mean)
    plt.title('Average Monthly Salary by Gender')

    plt.tight_layout()

    # Save the plot to a BytesIO object for streaming response
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")
