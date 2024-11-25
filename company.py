import streamlit as st
import os

def company_page():
    # Add custom CSS for styling
    st.markdown("""
    <style>
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .team-section {
        background-color: rgba(0,0,0,0.5);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add back button at the top
    if st.button("← Back to Main Page"):
        st.session_state.page = "main"
        st.experimental_rerun()
    
    # Title with styling
    st.markdown("""
    <div class="team-section">
        <h1 style='text-align: center; color: white;'>Meet Our Team</h1>
        <p style='text-align: center; color: white;'>The brilliant minds behind GeoSeg's innovation</p>
    </div>
    """, unsafe_allow_html=True)

    # Directory for team images
    images_dir = "team_images"

    # Define the team members with their roles and descriptions
    team = [
        {
            "name": "Malusi Ngubane",
            "role": "Chief Executive Officer",
            "description": "Leading GeoSeg's vision and strategy",
            "image": "malusi.jpg"
        },
        {
            "name": "Lebogang Mberu",
            "role": "Chief Technology Officer",
            "description": "Driving technological innovation",
            "image": "Lebogang.jpg"
        },
        {
            "name": "Mieke Spaans",
            "role": "Head of Research",
            "description": "Advancing our AI capabilities",
            "image": "Mieke Spaans.jpg"
        },
        {
            "name": "Sithabiseni Mtshali",
            "role": "Data Engineer",
            "description": "Building robust Data Engineering solutions",
            "image": "Sithabiseni Mtshali.png"
        },
        {
            "name": "Kamogelo Nkwana",
            "role": "Data Scientist",
            "description": "Creating intelligent algorithms",
            "image": "kamo.jpg"
        },
        {
            "name": "Frank Wilson",
            "role": "Product Manager",
            "description": "Shaping product strategy",
            "image": "frank.jpg"
        },
    ]

    # Display team member cards in a grid
    cols = st.columns(3)
    for i, member in enumerate(team):
        with cols[i % 3]:
            st.markdown("""
            <div style='background-color: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px; margin: 10px 0;'>
            """, unsafe_allow_html=True)
            
            image_path = os.path.join(images_dir, member["image"])
            if os.path.exists(image_path):
                st.image(image_path, caption="", use_column_width=True)
            else:
                st.image("https://via.placeholder.com/150", caption="", use_column_width=True)
            
            st.markdown(f"""
            <div style='text-align: center; color: white;'>
                <h3>{member['name']}</h3>
                <h4 style='color: #ff4b4b;'>{member['role']}</h4>
                <p>{member['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    # Add "Join Our Team" section at the bottom
    st.markdown("""
    <div class='team-section' style='margin-top: 30px; text-align: center;'>
        <h2>Join Our Team</h2>
        <p>We're always looking for talented individuals to join our mission of revolutionizing geospatial technology.</p>
    </div>
    """, unsafe_allow_html=True)
