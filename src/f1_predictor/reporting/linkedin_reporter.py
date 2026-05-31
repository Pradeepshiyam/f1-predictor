import plotly.graph_objects as go
import pandas as pd
import os

def generate_prediction_chart(predictions_df, gp_name, output_path="outputs/linkedin_report.png"):
    """
    Creates a premium-looking chart for LinkedIn.
    predictions_df should have columns: ['Driver', 'Probability']
    """
    # Sort by probability
    predictions_df = predictions_df.sort_values(by="Probability", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=predictions_df['Probability'],
        y=predictions_df['Driver'],
        orientation='h',
        marker=dict(
            color=predictions_df['Probability'],
            colorscale='Viridis',
            line=dict(color='rgba(255, 255, 255, 1.0)', width=1)
        )
    ))

    fig.update_layout(
        title=f"F1 Win Probability: {gp_name}",
        xaxis_title="Probability of Winning",
        yaxis_title="Driver",
        template="plotly_dark",
        font=dict(family="Arial, sans-serif", size=14, color="white"),
        margin=dict(l=100, r=20, t=50, b=50),
        paper_bgcolor='rgba(10, 10, 30, 1)',
        plot_bgcolor='rgba(10, 10, 30, 1)',
    )

    # Save as static image (requires kaleido)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    # Show it in the browser for now
    fig.show()
    print(f"Prediction chart generated for {gp_name}")

if __name__ == "__main__":
    # Sample data for testing
    sample_data = pd.DataFrame({
        'Driver': ['Verstappen', 'Norris', 'Leclerc', 'Hamilton', 'Sainz'],
        'Probability': [0.45, 0.22, 0.15, 0.10, 0.08]
    })
    generate_prediction_chart(sample_data, "Sample Grand Prix")
