import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
def load_data():
    df = pd.read_csv("results.csv")
    df = df.dropna()
    return df

# Preprocess data and train model
def preprocess_data(df):
    selected_columns = df[['home_team', 'away_team', 'home_score', 'away_score']]
    selected_columns = selected_columns.rename(columns={'home_team': 'team_one', 'away_team': 'team_two'})

    def match_result(row):
        if row['home_score'] > row['away_score']:
            return 1  # Home team wins
        elif row['home_score'] < row['away_score']:
            return -1  # Home team loses
        else:
            return 0  # Draw

    df['result'] = df.apply(match_result, axis=1)
    selected_columns['result'] = selected_columns.apply(match_result, axis=1)

    unique_teams = pd.concat([selected_columns['team_one'], selected_columns['team_two']]).unique()

    le = LabelEncoder()
    le.fit(unique_teams)

    selected_columns['team_one_encoded'] = le.transform(selected_columns['team_one'])
    selected_columns['team_two_encoded'] = le.transform(selected_columns['team_two'])

    X = selected_columns[['team_one_encoded', 'team_two_encoded']]
    y = selected_columns['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, le, selected_columns


# Streamlit app
def main():
    st.set_page_config(
        page_title="Football Match Predictor",
  
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title('Football Match Predictor')
    
    # Load data and train model
    df = load_data()
    model, scaler, le, selected_columns = preprocess_data(df)

    # User input for team names
    team_one = st.text_input("Enter the name of team one:")
    team_two = st.text_input("Enter the name of team two:")
    
    if st.button('Predict'):
        if team_one and team_two:
            team_one_encoded = le.transform([team_one])[0]
            team_two_encoded = le.transform([team_two])[0]
            features = scaler.transform([[team_one_encoded, team_two_encoded]])
            prediction = model.predict(features)[0]
            
            if prediction == -1:
                st.write(f'{team_one} will lose against {team_two}')
            elif prediction == 0:
                st.write(f'{team_one} and {team_two} will draw')
            else:
                st.write(f'{team_one} will win against {team_two}')

    st.subheader('Charts')

    # Get all unique teams from the dataset
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()

    # Chart 1: Count of matches for each team
    st.subheader('Count of matches for each team')

    # Dropdown to select a team
    selected_team = st.selectbox('Select a team', ['All'] + list(all_teams))

    # Filter data based on selected team
    if selected_team == 'All':
        filtered_df = df  # Show all matches
    else:
        filtered_df = df[(df['home_team'] == selected_team) | (df['away_team'] == selected_team)]

    # Calculate counts for the selected team
    if selected_team == 'All':
        team_counts = pd.concat([filtered_df['home_team'], filtered_df['away_team']]).value_counts()
    else:
        home_counts = filtered_df[filtered_df['home_team'] == selected_team]['home_team'].value_counts()
        away_counts = filtered_df[filtered_df['away_team'] == selected_team]['away_team'].value_counts()
        team_counts = home_counts.add(away_counts, fill_value=0).astype(int)

    # Display the bar chart for the selected team
    st.bar_chart(team_counts)

    # Chart 2: Win/draw/loss record for the selected team
    st.subheader('Win/Draw/Loss record for selected team')

    if selected_team != 'All':
        team_matches = filtered_df[(filtered_df['home_team'] == selected_team) | (filtered_df['away_team'] == selected_team)]

        win_count = len(team_matches[team_matches['result'] == 1])
        draw_count = len(team_matches[team_matches['result'] == 0])
        loss_count = len(team_matches[team_matches['result'] == -1])

        labels = ['Wins', 'Draws', 'Losses']
        values = [win_count, draw_count, loss_count]

        # Plot pie chart using matplotlib
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Display the pie chart using st.pyplot
        st.pyplot(fig)

        # Display numbers below the pie chart
        st.write(f"Number of Wins: {win_count}")
        st.write(f"Number of Draws: {draw_count}")
        st.write(f"Number of Losses: {loss_count}")

if __name__ == '__main__':
    main()
