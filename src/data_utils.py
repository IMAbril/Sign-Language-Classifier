import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd 

def plot_statistics_grid(data_grouped, label_mapping, title, cmap='gray'):
    """
    Plots a 4x6 grid of images based on grouped pixel statistics.
    """
    plt.figure(figsize=(16, 10))
    
    for i, (label_idx, letter) in enumerate(label_mapping.items()):
        # Reshape the 784 flat pixels back to 28x28 spatial matrix
        pixels = data_grouped.loc[label_idx].values.reshape(28, 28)
        
        plt.subplot(4, 6, i + 1)
        plt.imshow(pixels, cmap=cmap)
        plt.title(f"Letra: {letter}", fontsize=12)
        plt.axis('off')
        
    plt.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

def compare_similar_letters(letter1, letter2, mean_data, label_mapping, title='Análisis de Confusión Potencial: M vs N'):
    """
    Plots the average images of two letters and their absolute difference.
    """
    # Find the corresponding label indices
    idx1 = [k for k, v in label_mapping.items() if v == letter1][0]
    idx2 = [k for k, v in label_mapping.items() if v == letter2][0]
    
    # Get arrays
    img1 = mean_data.loc[idx1].values.reshape(28, 28)
    img2 = mean_data.loc[idx2].values.reshape(28, 28)
    
    # Calculate absolute difference pixel by pixel
    diff_img = np.abs(img1 - img2)
    
    # Visualization setup
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(f"Promedio: Letra '{letter1}'", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(f"Promedio: Letra '{letter2}'", fontsize=14)
    axes[1].axis('off')
    
    # Difference Heatmap
    im = axes[2].imshow(diff_img, cmap='hot')
    axes[2].set_title(f"Diferencia Absoluta ('{letter1}' vs '{letter2}')", fontsize=14)
    axes[2].axis('off')
    
    # Add colorbar to understand the magnitude of difference
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Diferencia Absoluta', rotation=270, labelpad=15)
    
    plt.suptitle(f"{title}", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()



def plot_model_performance(df, model_name, x_axis_param, hue_param='PCA', title_suffix=""):
    """
    Generate a dual-axis plot to visualize model performance metrics.
    
    """
    # Data preparation: Ensure categorical types for cleaner plotting
    plot_data = df.copy()
    plot_data[x_axis_param] = plot_data[x_axis_param].astype(str)
    plot_data[hue_param] = plot_data[hue_param].astype(str)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # --- Left Axis: F1-SCORE (Stability and Variance) ---
    # Boxplot shows the distribution across cross-validation folds
    sns.boxplot(data=plot_data, x=x_axis_param, y='macro_f1', hue=hue_param, 
                palette='Blues', ax=ax1, dodge=True)
    
    # Stripplot adds visibility to individual fold results
    sns.stripplot(data=plot_data, x=x_axis_param, y='macro_f1', hue=hue_param, 
                  dodge=True, palette='dark:black', alpha=0.3, ax=ax1, legend=False)

    ax1.set_xlabel(f"Hiperparámetro: {x_axis_param}", fontsize=12)
    ax1.set_ylabel("Macro F1-Score", fontsize=12, color='navy')
    
    # Set limits for better visualization of perfect scores
    ax1.set_ylim(plot_data['macro_f1'].min() - 0.01, 1.01)

    # --- Right Axis: COMPUTATIONAL EFFICIENCY (Training Time) ---
    ax2 = ax1.twinx()
    sns.pointplot(data=plot_data, x=x_axis_param, y='fit_time', hue=hue_param, 
                  palette='Reds', markers='D', linestyles='--', ax=ax2, errorbar=None, dodge=0.5)

    ax2.set_ylabel("Tiempo de Entrenamiento (s)", fontsize=12, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    # Legend management
    ax1.legend(title=f"{hue_param} Componentes", loc='upper left')
    if ax2.get_legend(): ax2.get_legend().remove()

    plt.title(f"{model_name} Performance Analisis {title_suffix}", fontsize=15)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

def get_best_representative(df, model_name, best_params_dict):
    """
    Extract the best performing configuration for a specific model 
    to be used in inter-model comparison.
    """
    mask = (df['model'] == model_name)
    for param, value in best_params_dict.items():
        mask &= (df[param] == value)
    
    rep_df = df[mask].copy()
    
    # Logging key metrics for the selected representative
    print(f"--- Modelo Óptimo: {model_name} ---")
    print(f"Fit Time Promedio: {rep_df.fit_time.mean():.4f}s")
    print(f"Macro F1 Promedio: {rep_df.macro_f1.mean():.4f}\n")
    
    return rep_df

def reconstruct_images(X, scaler, pca):
    """
    Reconstruct images using the PCA subspace.
    """
    
    # Scale data
    X_scaled = scaler.transform(X)
    
    # Project to PCA space
    X_pca = pca.transform(X_scaled)
    
    # Reconstruct in scaled space
    X_reconstructed_scaled = pca.inverse_transform(X_pca)
    
    # Return to original space
    X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)
    
    return X_reconstructed

def get_top_confusions(cm, labels, top_n=5):
    """
    Extracts the pairs of classes that are most frequently mistaken for each other.
    """
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'Real': labels[i],
                    'Predicha': labels[j],
                    'Errores': cm[i, j]
                })
    
    # Sort by number of errors descending
    confusions_df = pd.DataFrame(confusions).sort_values(by='Errores', ascending=False)
    return confusions_df.head(top_n)