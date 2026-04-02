### Text-based visualization with colored boxes (like highlighted sentence)
from IPython.display import HTML, display
import matplotlib as mpl
import matplotlib.pyplot as plt  
from matplotlib.colors import Normalize as Norm, LogNorm as Log_Norm
import torch
import numpy as np

def natural_language_attribution_plotter(decoded_tokens, grad_idx, scores, mode, loss_position, log_color = False, style = "A"):
    # cmap = plt.get_cmap('coolwarm')
    # cmap = plt.get_cmap('viridis')
    # cmap = plt.get_cmap('Greens')
    cmap = plt.get_cmap('Blues')

    optimized_tokens = [decoded_tokens[idx] for idx in grad_idx]

    tick_label_text = optimized_tokens.copy()
                
    grad_magnitude = scores

    grad_magnitude[grad_idx.index(loss_position+1)] = max(grad_magnitude[:grad_idx.index(loss_position+1)])

    # Normalize grad_magnitude for color mapping
    log_norm = Log_Norm(vmin=grad_magnitude.min(), vmax=grad_magnitude.max())
    norm = Norm(vmin=grad_magnitude.min(), vmax=grad_magnitude.max())

    if mode == 'Temperature' or mode == 'Fisher':
        tick_label_text[grad_idx.index(loss_position+1)] = "&lt;predicted distribution&gt;"


    bar_idx = grad_idx.index(loss_position+1)

    if log_color:
        colors = cmap(log_norm(grad_magnitude))
    else:
        colors = cmap(norm(grad_magnitude))

    # Build HTML with colored boxes around each word, using bold font
    def rgba_to_css(rgba):
        """Convert matplotlib RGBA to CSS rgba string"""
        return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]:.2f})"

    def get_text_color(bg_rgba):
        """Return white or black text based on background luminance"""
        luminance = 0.299 * bg_rgba[0] + 0.587 * bg_rgba[1] + 0.114 * bg_rgba[2]
        return "white" if luminance < 0.5 else "black"

    html_parts = []
    for i, (token, color) in enumerate(zip(tick_label_text, colors)):
        bg_color = rgba_to_css(color)
        text_color = get_text_color(color)
        
        # Special styling for target token
        if i == bar_idx:
            # bg_color = "black"
            bg_color = "red"
            text_color = "white"
        
        
        
        if style == "A":
            display_token = token.strip() or "·"
            html_parts.append(
                f'<span style="'
                f'background-color: {bg_color}; '
                f'color: {text_color}; '
                f'padding: 0px 9px; '
                f'margin: 2px; '
                f'border-radius: 8px; '
                f'font-family: monospace; '
                f'font-size: 16px; '
                f'display: inline-block; '
                f'font-weight: bold;'   # <-- bold font here
                f'">{display_token}</span>'
            )
        elif style == "B":
            display_token = token
            html_parts.append(
            f'<span style="'
            f'background-color: {bg_color}; '
            f'color: {text_color}; '
            f'padding: 0px 0px; '
            f'margin: 0px; '
            f'border-radius: 0px; '
            f'font-family: monospace; '
            f'font-size: 16px; '
            f'display: inline-block; '
            f'font-weight: bold;'   # <-- bold font here
            f'white-space: pre;">{display_token}</span>'
        )

    html_str = f'''
    <div style="
        background: white; 
        padding: 20px; 
        border-radius: 8px; 
        line-height: 2.2;
        # max-width: 250%;
        max-width: 600px;
        word-break: break-word;
    ">
        {"".join(html_parts)}
    </div>
    '''

    display(HTML(html_str))
    


    # Adjust the figure size to better fit the colorbar and avoid cutoff
    fig_bar, ax_bar = plt.subplots(figsize=(0.4, 2.2), dpi=250)
    fig_bar.subplots_adjust(left=0.3, right=0.7, bottom=0.1, top=0.9)

    cbar = mpl.colorbar.ColorbarBase(
        ax_bar,
        cmap=cmap,
        norm=log_norm if log_color else norm,
        orientation='vertical'
    )
    cbar.set_label('Influence', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    plt.show()
    
def natural_language_pred_plotter(probs, tokenizer, k = 7):
    # Get top k probabilities and their indices
    top_probs, top_indices = torch.topk(probs, k)
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,1.5), dpi=100)
    bars = plt.bar(range(k), top_probs.cpu().numpy(), tick_label=top_tokens, color='red')
    plt.xticks(rotation=90, fontweight='bold', fontsize=18)
    # plt.xlabel('Token',rotation=180, fontweight='bold', fontsize=12)
    plt.ylabel('Probability', fontweight='bold', fontsize=16,rotation=90)
    plt.yticks(rotation=90, fontweight='bold', fontsize=12)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=2))  # increase number of ticks

    # plt.title(f'Top {k} Tokens in Softmax of Logits')
    plt.show()

def numerical_pred_plotter(true_token, logits, tokenizer, loss_position, k = 30):
    logit_vector = logits[loss_position].detach()             # shape: [vocab_size]
    # Compute softmax probabilities
    probs = torch.softmax(logit_vector, dim=-1)

    # Get top k probabilities and their indices
    top_probs, top_indices = torch.topk(probs, k)
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    # Reorder top_probs and top_tokens by int(top_tokens)
    sorted_with_probs = sorted(zip(top_tokens, top_probs), key=lambda x: int(x[0]))
    top_tokens, top_probs = zip(*sorted_with_probs)
    top_probs = torch.stack(top_probs) if isinstance(top_probs[0], torch.Tensor) else torch.tensor(top_probs)


    plt.figure(figsize=(6,2.5), dpi=200)
    bars = plt.bar(range(k), top_probs.cpu().numpy(), tick_label=top_tokens, color = 'blue')
    # Color the bar at true_target in red
    for i, token in enumerate(top_tokens):
        if token == true_token:
            bars[i].set_color('red')
            bars[i].set_label("true token")
            break
    else:
        print("true token not found")
    plt.xticks(rotation=90, fontweight='bold', fontsize=13)
    plt.ylabel('Probability', fontweight='bold', fontsize=22,rotation=90)
    plt.yticks(rotation=90, fontweight='bold', fontsize=12)
    # plt.xlabel('Token')
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=2))  # increase number of ticks
    plt.legend()
    plt.show()        
    

import numpy as np
from matplotlib.colors import Normalize as Norm, LogNorm as Log_Norm
import matplotlib.pyplot as plt  

def numerical_attribution_plotter(input_string, scores, grad_idx, front_pad, loss_position,
                                  label_font_size = 12, log_scale_score = False, show_bars = True):
    # Hyperparameters
    surpress_last_value = 1

    # Axes color hyperparameters
    ax1_color = 'blue'
    ax2_color = 'red'

    ax1_color = np.array([10, 110, 230])/256
    ax2_color = np.array([230, 20, 20])/256

    x_labels = [x-front_pad for x in grad_idx]

    scores[grad_idx.index(loss_position+1)] = max(scores)
        
    int_list = [int(x) for x in input_string.split(',')]
    if surpress_last_value:
        int_list = int_list[:-1]


    plt.figure(figsize=(7, 1.6), dpi=200)

    if show_bars:
        grad_mag_np = np.asarray(scores).copy()
        if log_scale_score:
            # Log scale requires positive y; clip to small epsilon so bars remain visible
            grad_mag_np = np.maximum(grad_mag_np, 1e-12)
        bars = plt.bar(
            range(len(grad_mag_np)),
            grad_mag_np,
            tick_label=x_labels,
            color=ax1_color,
            linewidth=0.5,        # Smaller bar edge
            edgecolor='black',     # Remove edge color to help overlap
            width=1.0,            # Make bars full width; may cause overlap at dense ticks
            alpha=0.9             # Let bars visually overlap via alpha (transparency)
        )
        target_bar_index = grad_idx.index(loss_position+1)
        bars[target_bar_index].set_color('red')
        bars[target_bar_index].set_width(1.1)  # make this specific bin larger

    ax = plt.gca()
    ax2 = ax.twinx()

    under_sample_rate = 50
    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()
    # Only keep every Nth tick and label
    new_xticks = [tick for i, tick in enumerate(xticks) if i % under_sample_rate == 0]
    new_xticklabels = [label.get_text() for i, label in enumerate(xticklabels) if i % under_sample_rate == 0]
    # ax.set_xticks(new_xticks+[grad_idx.index(loss_position+1)])
    ax.set_xticks(new_xticks+[grad_idx.index(loss_position+1)])


    # Matplotlib's set_xticklabels normally expects strings or Text instances, but to ensure color:
    # Set all to black but override the last in-place after drawing
    ax.set_xticklabels(new_xticklabels + ["target"], fontsize=label_font_size)
    # Suppress the tick but keep only the label for the last xtick
    # Get all xticks and xticklabels
    xticks = ax.get_xticks()
    labels = ax.get_xticklabels()

    # Hide the tick for the last xtick but keep its label, and color it red
    if len(xticks) > 0 and len(labels) > 0:
        last_tick_index = -1
        # Set the label color to red
        labels[last_tick_index].set_color('red')
        # Hide the tick by overlaying an 'empty' tick at its position
        tick_locs = xticks.tolist()
        if len(tick_locs) > 1:
            tick_locs_no_last = tick_locs[:-1]
            label_texts_no_last = [l.get_text() for l in labels[:-1]]
            # Set ticks and labels, then add the last label only manually as text
            ax.set_xticks(tick_locs_no_last)
            ax.set_xticklabels(label_texts_no_last, fontsize=label_font_size)
            # add the last label manually
            if not log_scale_score:
                ax.text(tick_locs[-1], ax.get_yticks()[0] - 0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0]), labels[-1].get_text(),
                        color='red', fontsize=label_font_size, va='top', ha='center', fontweight='bold')
            else:
                # Log scale: use axes coords for y so position is below plot regardless of scale
                ax.text(tick_locs[-1], -0.08, labels[-1].get_text(), transform=ax.get_xaxis_transform(),
                        color='red', fontsize=label_font_size, va='top', ha='center', fontweight='bold')

    ax2.scatter(
        range(len(int_list)), int_list,
        c=ax2_color, marker='o', s=13, alpha=0.9
    )
    ax2.plot(
        range(len(int_list)), int_list,
        c=ax2_color, linewidth=1.5, alpha=0.5
    )
    ax2.tick_params(axis='y', colors=ax2_color, labelsize=label_font_size)
    ax.tick_params(axis='y', colors=ax1_color, labelsize=label_font_size)

    ax.set_xlabel('Token position index', fontsize=label_font_size, fontweight='bold')
    ax.set_ylabel('Influence', labelpad=2, color=ax1_color, fontsize=label_font_size, fontweight='bold')
    ax2.set_ylabel('Token value', labelpad=2, color=ax2_color, fontsize=label_font_size, fontweight='bold')

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
    ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
    ax.minorticks_on()
    # plt.tight_layout()
    if log_scale_score:
        ax.set_yscale('log')

    plt.show()