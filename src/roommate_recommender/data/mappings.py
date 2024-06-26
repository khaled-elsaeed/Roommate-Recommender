# mappings.py

ordinal_mappings = {
    'Bedtime_Preference': {'Before midnight': 0, 'Around midnight': 1, 'at any time': 2, 'After midnight': 3},
    'Wake_Up_Time_Preference': {'Very early': 0, 'Somewhat early': 1, 'at any time': 2, 'Late': 3, 'As late as possible': 4},
    'Planned_Study_Time': {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Late night': 3},
    'Private_Time_Requirements': {'Very little': 0, 'Some': 1, 'Significant': 2},
    'Guest_Frequency_Preference': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Frequently': 4},
    'Ideal_Study_Environment_Description_encoded': {'Background noise': 0, 'High noise tolerance': 1, 'Very quiet': 2},
    'Attitude_towards_Borrowing_Sharing': {'Ask before sharing': 0, 'Prefer personal items': 1, 'Share freely': 2},
    'Description_of_Personal_Room_At_Home_encoded': {'Clean and organized': 0, 'Cluttered': 1, 'Disorganized': 2, 'Neat': 3},
    'Desired_Room_Attributes_encoded': {'Social & quiet': 0, 'Social gathering': 1, 'Study oriented': 2},
    'Study_Time_Preference': {'Background noise': 0, 'No preference': 1, 'Total quiet': 2},
    'Conflict_Handling_Method_encoded': {'Avoid conflict': 0, 'Blunt': 1, 'Calm': 2, 'Hint jokingly': 3},
    'Communication_Preference_with_Roommate': {'Face-to-face': 0, 'Notes/letters': 1, 'Text or messaging apps': 2},
    'Age_normalized': {}  # Placeholder for Age_normalized
}

nominal_mappings = {
    'Faculty': {'Business': 0, 'Computer Science': 1, 'Dentistry': 2, 'Engineering': 3, 'Health Sciences': 4,
                'Medicine': 5, 'Nursing': 6, 'Pharmacy': 7, 'Science': 8, 'Textile Engineering': 9},
    'Attitude_towards_Roommate_Smoking': {'Maybe': 0, 'No': 1, 'Yes': 2},
}
