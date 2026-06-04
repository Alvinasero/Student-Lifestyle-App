
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model, scaler, and data
@st.cache_resource
def load_resources():
    model = joblib.load('stacking_model.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    df = pd.read_csv('student_lifestyle_dataset.csv')
    # Ensure Academic_Performance is created if not already in df
    if 'Academic_Performance' not in df.columns:
        gpa_bins = [0, 2.5, 3.0, 3.5, 4.0]
        gpa_labels = ['Poor', 'Fair', 'Good', 'Excellent']
        df['Academic_Performance'] = pd.cut(df['GPA'], bins=gpa_bins, labels=gpa_labels, include_lowest=True)

    evaluation_summary = joblib.load('evaluation_summary.pkl')
    feature_importance_df = joblib.load('feature_importance_df.pkl')

    return model, scaler, df, evaluation_summary, feature_importance_df

model, scaler, df, evaluation_summary, feature_importance_df = load_resources()

# Reverse mapping for stress levels
stress_level_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}

st.sidebar.title('Navigation')
selection = st.sidebar.radio('Go to', ['Prediction', 'Data Exploration', 'Model Evaluation', 'Student Planner'])


# --- Prediction Page ---
if selection == 'Prediction':
    st.title('Student Stress Level Prediction')
    st.write('Enter the student\'s lifestyle data to predict their stress level.')

    # Input fields for features
    study_hours = st.slider('Study Hours Per Day', 0.0, 10.0, 7.0)
    extracurricular_hours = st.slider('Extracurricular Hours Per Day', 0.0, 5.0, 2.0)
    sleep_hours = st.slider('Sleep Hours Per Day', 0.0, 12.0, 7.0)
    social_hours = st.slider('Social Hours Per Day', 0.0, 5.0, 2.0)
    physical_activity_hours = st.slider('Physical Activity Hours Per Day', 0.0, 10.0, 3.0)
    gpa = st.slider('GPA', 0.0, 4.0, 3.0, step=0.01)

    if st.button('Predict Stress Level'):
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame([[study_hours, extracurricular_hours, sleep_hours,
                                    social_hours, physical_activity_hours, gpa]],
                                  columns=['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day',
                                           'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day',
                                           'Physical_Activity_Hours_Per_Day', 'GPA'])

        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction_encoded = model.predict(scaled_input)[0]

        # Map prediction back to original stress level
        predicted_stress_level = stress_level_mapping[prediction_encoded]

        st.success(f'The predicted stress level is: **{predicted_stress_level}**')

# --- Data Exploration Page ---
elif selection == 'Data Exploration':
    st.title('Data Exploration')
    st.write('Explore the distribution and relationships within the student lifestyle dataset.')

    st.subheader('Dataset Overview')
    st.write(f'Shape of the dataset: {df.shape}')
    # Make df.head() editable
    edited_df_head = st.data_editor(df.head(), key='dataset_overview_editor')
    st.caption("Note: Edits to this table are temporary and do not modify the original dataset.")

    st.subheader('Stress Level Distribution')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(y='Stress_Level', data=df, palette='muted', ax=axes[0])
    axes[0].set_title('Distribution of Stress Level')
    df['Stress_Level'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('muted'), startangle=90, explode=[0.05]*df['Stress_Level'].nunique(), ax=axes[1])
    axes[1].set_title('Percentage Distribution of Stress Level')
    axes[1].set_ylabel('')
    st.pyplot(fig)

    st.subheader('Academic Performance Distribution')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(y='Academic_Performance', data=df, palette='muted', ax=axes[0], order=df['Academic_Performance'].value_counts().index)
    axes[0].set_title('Distribution of Academic Performance')
    df['Academic_Performance'].value_counts(dropna=False).plot.pie(autopct='%1.1f%%', colors=sns.color_palette('muted'), startangle=90, explode=[0.05]*df['Academic_Performance'].nunique(), pctdistance=0.85, ax=axes[1])
    axes[1].set_title('Percentage Distribution of Academic Performance')
    axes[1].set_ylabel('')
    st.pyplot(fig)

    st.subheader('Numerical Features Distribution')
    numerical_cols = ['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']
    fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(10, 4 * len(numerical_cols)))
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette('muted')[i%len(sns.color_palette('muted'))])
        axes[i].set_title(f'Distribution of {col.replace("_", " ")}')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))
    numerical_columns_for_corr = df.select_dtypes(include=np.number)
    sns.heatmap(numerical_columns_for_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap of Numerical Features')
    st.pyplot(fig)

# --- Model Evaluation Page ---
elif selection == 'Model Evaluation':
    st.title('Model Evaluation')
    st.write('Review the performance metrics and insights from the trained machine learning model.')

    st.subheader('Model Performance Summary')
    # Make evaluation_summary editable
    edited_evaluation_summary = st.data_editor(evaluation_summary, key='evaluation_summary_editor')
    st.caption("Note: Edits to this table are temporary and do not modify the original evaluation summary file.")

    st.subheader('Feature Importance')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='muted', ax=ax)
    ax.set_title('Feature Importance - Random Forest Base Model')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    st.pyplot(fig)

    st.subheader('Predicted Stress Level Distribution on Test Set')
    st.write("*(Note: The distribution below shows the overall 'Stress_Level' from the original dataset, ")
    st.write(" as a direct `y_pred_mapped` from the test set was not saved for this app. ")
    st.write("In a full deployment, `y_pred_mapped` would also be saved and loaded.)*")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(y='Stress_Level', data=df, palette='muted', ax=ax)
    ax.set_title('Distribution of Overall Stress Levels (as a proxy for predicted)')
    ax.set_xlabel('Count')
    ax.set_ylabel('Stress Levels')
    st.pyplot(fig)

# --- Student Planner Page ---
elif selection == 'Student Planner':
    st.title('Student Daily Planner')
    st.write('Create a personalized daily schedule based on your preferences.')

    st.subheader('Input Your Daily Activity Preferences')

    total_hours_day = 24.0
    current_allocated_hours = 0.0

    sleep_hours = st.slider('Sleep Hours', 5.0, 12.0, 8.0, step=0.5)
    current_allocated_hours += sleep_hours

    study_hours_max = max(0.0, total_hours_day - current_allocated_hours - 2) # Leave 2 hours for buffer
    study_hours = st.slider('Study Hours', 0.0, study_hours_max, min(7.0, study_hours_max), step=0.5)
    current_allocated_hours += study_hours

    meal_hours_max = max(0.0, total_hours_day - current_allocated_hours - 1) # Leave 1 hour for buffer
    meal_hours = st.slider('Meal Hours', 1.0, meal_hours_max, min(2.0, meal_hours_max), step=0.5)
    current_allocated_hours += meal_hours

    physical_activity_hours_max = max(0.0, total_hours_day - current_allocated_hours - 0.5)
    physical_activity_hours = st.slider('Physical Activity Hours', 0.0, physical_activity_hours_max, min(1.5, physical_activity_hours_max), step=0.5)
    current_allocated_hours += physical_activity_hours

    social_hours_max = max(0.0, total_hours_day - current_allocated_hours - 0.5)
    social_hours = st.slider('Social/Leisure Hours', 0.0, social_hours_max, min(2.0, social_hours_max), step=0.5)
    current_allocated_hours += social_hours

    remaining_hours = max(0.0, total_hours_day - current_allocated_hours)

    st.write(f"*Remaining unallocated hours: {remaining_hours:.1f}*")
    st.caption("Adjust sliders to fit your day. Hours are dynamically limited based on previous selections to ensure they don't exceed 24 hours.")

    if st.button('Generate Schedule'):
        st.subheader('Your Personalized Daily Schedule')
        st.write("Here's a sample daily plan based on your inputs. Feel free to adjust based on your specific needs and class schedule!")

        schedule = []
        current_time = 0.0 # Representing hours from midnight

        # Sleep (e.g., 10 PM to 6 AM for 8 hours sleep)
        schedule.append(f"22:00 - {(22 + sleep_hours) % 24:04.2f}: Sleep ({sleep_hours:.1f} hours)")
        current_time = (22 + sleep_hours) % 24 # Assuming sleep starts at 10 PM for simplicity

        # Morning Routine / Breakfast
        schedule.append(f"{current_time:04.2f} - {(current_time + 1):04.2f}: Morning Routine / Breakfast (1.0 hour)")
        current_time += 1.0

        # Study Hours - distribute throughout the day
        if study_hours > 0:
            study_block = study_hours / 2 # Two study blocks
            schedule.append(f"{current_time:04.2f} - {(current_time + study_block):04.2f}: Study Block 1 ({study_block:.1f} hours)")
            current_time += study_block

        # Lunch
        schedule.append(f"{current_time:04.2f} - {(current_time + meal_hours/2):04.2f}: Lunch ({meal_hours/2:.1f} hours)")
        current_time += meal_hours/2

        if study_hours > study_block: # If there's a second study block
            schedule.append(f"{current_time:04.2f} - {(current_time + study_block):04.2f}: Study Block 2 ({study_block:.1f} hours)")
            current_time += study_block

        # Physical Activity
        if physical_activity_hours > 0:
            schedule.append(f"{current_time:04.2f} - {(current_time + physical_activity_hours):04.2f}: Physical Activity ({physical_activity_hours:.1f} hours)")
            current_time += physical_activity_hours

        # Dinner
        schedule.append(f"{current_time:04.2f} - {(current_time + meal_hours/2):04.2f}: Dinner ({meal_hours/2:.1f} hours)")
        current_time += meal_hours/2

        # Social/Leisure Hours
        if social_hours > 0:
            schedule.append(f"{current_time:04.2f} - {(current_time + social_hours):04.2f}: Social/Leisure Time ({social_hours:.1f} hours)")
            current_time += social_hours

        # Remaining time (flexible/buffer)
        if remaining_hours > 0:
            schedule.append(f"{current_time:04.2f} - {(current_time + remaining_hours):04.2f}: Flexible/Buffer Time ({remaining_hours:.1f} hours)")

        for item in schedule:
            st.write(f"- {item}")
