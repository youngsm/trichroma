from tqdm import tqdm
import torch
import plotly.graph_objects as go
from chroma.event import SURFACE_DETECT, NO_HIT, NAN_ABORT, SURFACE_ABSORB, BULK_ABSORB
import numpy as np

def plot_geometry(event=None, g=None, plot_tracks=False, track_colorscale='Viridis', bomb_positions=None):
    fig = go.Figure()

    if g is not None:
        solids = g.solids
        for i, solid in enumerate(tqdm(solids)):
            colors = solid.color  # (N,) in hex (uint16)
            triangles = solid.mesh.triangles  # (N, 3)
            vertices = solid.mesh.vertices  # (M, 3), M<=N

            # Convert to PyTorch tensors on GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            vertices_torch = torch.tensor(vertices, device=device, dtype=torch.float32)
            try:
                pos = torch.tensor(g.solid_displacements[i], device=device, dtype=torch.float32)
                rot = torch.tensor(g.solid_rotations[i], device=device, dtype=torch.float32)
            except Exception:
                pos = None
                rot = None

            # Convert hex colors to RGB format for plotly
            rgb_colors = []
            for color in colors:
                # Extract RGB components from uint16 hex
                r = ((color >> 10) & 0x1F) / 31.0
                _g = ((color >> 5) & 0x1F) / 31.0
                b = (color & 0x1F) / 31.0
                rgb_colors.append(f"rgb({int(r * 255)},{int(_g * 255)},{int(b * 255)})")

            # Create mesh3d for each solid
            i_vertices = []
            j_vertices = []
            k_vertices = []

            for triangle in triangles:
                i_vertices.append(triangle[0])
                j_vertices.append(triangle[1])
                k_vertices.append(triangle[2])

            # Perform displacements using GPU
            if rot is not None:
                vertices_torch = torch.matmul(vertices_torch, -rot)
            if pos is not None:
                vertices_torch = vertices_torch + pos

            # Convert back to numpy for plotting
            vertices = vertices_torch.cpu().numpy()

            solid_name = None
            if hasattr(g, 'solid_names') and i < len(g.solid_names) and g.solid_names[i] is not None:
                solid_name = g.solid_names[i]
            name = solid_name if solid_name else f"Solid {i}"
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=i_vertices,
                    j=j_vertices,
                    k=k_vertices,
                    facecolor=rgb_colors,
                    opacity=0.1,
                    name=name,
                )
            )

        if hasattr(g, "wireplanes") and g.wireplanes:
            for idx, wp in enumerate(g.wireplanes):
                try:
                    origin = np.asarray(wp["origin"], dtype=np.float32)
                    u = np.asarray(wp["u"], dtype=np.float32)
                    v = np.asarray(wp["v"], dtype=np.float32)
                    umin = float(wp["umin"])
                    umax = float(wp["umax"])
                    vmin = float(wp["vmin"])
                    vmax = float(wp["vmax"])
                except Exception:
                    continue
                c0 = origin + u * umin + v * vmin
                c1 = origin + u * umax + v * vmin
                c2 = origin + u * umax + v * vmax
                c3 = origin + u * umin + v * vmax
                vertices_rect = np.vstack([c0, c1, c2, c3])
                i_idx, j_idx, k_idx = [0, 0], [1, 2], [2, 3]
                fig.add_trace(
                    go.Mesh3d(
                        x=vertices_rect[:, 0],
                        y=vertices_rect[:, 1],
                        z=vertices_rect[:, 2],
                        i=i_idx, j=j_idx, k=k_idx,
                        color="lightgray", opacity=0.2,
                        name=f"Wireplane {idx} (wall)",
                    )
                )

    if bomb_positions is not None:
        pos = np.asarray(bomb_positions, dtype=np.float64)
        if pos.ndim == 2 and pos.shape[1] == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=pos[:, 0],
                    y=pos[:, 1],
                    z=pos[:, 2],
                    mode="markers",
                    marker=dict(size=2, color="orange", opacity=0.8, symbol="square"),
                    name="Bomb positions",
                )
            )

    if event is not None:
        # Plot beginning photons
        fig.add_trace(
            go.Scatter3d(
                x=event.photons_beg.pos[:,0],
                y=event.photons_beg.pos[:,1],
                z=event.photons_beg.pos[:,2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.5
                ),
                name='Beginning Photons'
            )
        )

        # Plot ending photons with different colors based on their flags
        # Detected photons
        detected_mask = (event.photons_end.flags & SURFACE_DETECT) == SURFACE_DETECT
        if detected_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=event.photons_end.pos[detected_mask,0],
                    y=event.photons_end.pos[detected_mask,1],
                    z=event.photons_end.pos[detected_mask,2],
                    mode='markers',
                    marker=dict(
                        size=0.5,
                        color='green',
                        opacity=0.5,
                    ),
                    name='Detected Photons'
                )
            )

        # No hit photons
        nohit_mask = (event.photons_end.flags & NO_HIT) == NO_HIT
        if nohit_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=event.photons_end.pos[nohit_mask,0],
                    y=event.photons_end.pos[nohit_mask,1],
                    z=event.photons_end.pos[nohit_mask,2],
                    mode='markers',
                    marker=dict(
                        size=0.5,
                        color='black',
                        opacity=0.5
                    ),
                    name='No Hit Photons'
                )
            )

        # Absorbed photons (surface and bulk)
        surface_absorbed_mask = (((event.photons_end.flags & SURFACE_ABSORB) == SURFACE_ABSORB))
        if surface_absorbed_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=event.photons_end.pos[surface_absorbed_mask,0],
                    y=event.photons_end.pos[surface_absorbed_mask,1],
                    z=event.photons_end.pos[surface_absorbed_mask,2],
                    mode='markers',
                    marker=dict(
                        size=0.5,
                        color='red',
                        opacity=0.5
                    ),
                    name='Surface Absorbed Photons'
                )
            )

        bulk_absorbed_mask = (((event.photons_end.flags & BULK_ABSORB) == BULK_ABSORB))
        if bulk_absorbed_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=event.photons_end.pos[bulk_absorbed_mask,0],
                    y=event.photons_end.pos[bulk_absorbed_mask,1],
                    z=event.photons_end.pos[bulk_absorbed_mask,2],
                    mode='markers',
                    marker=dict(
                        size=0.5,
                        color='black',
                        opacity=0.5
                    ),
                    name='Bulk Absorbed Photons'
                )
            )


        if plot_tracks and hasattr(event, 'photon_tracks'):
            # group tracks by their end flags
            flag_groups = {
                'detected': SURFACE_DETECT,
                'no_hit': NO_HIT,
                'aborted': NAN_ABORT,
                'surface_absorbed': SURFACE_ABSORB,
                'bulk_absorbed': BULK_ABSORB
            }
            
            # track traces by flag type for legend grouping
            track_traces = {}
            
            # track wireplane hits separately
            wireplane_hit_indices = []
            wireplane_detected_indices = []

            min_t = np.min(event.photons_end.t)
            max_t = np.max(event.photons_end.t)
            
            for i, track in enumerate(event.photon_tracks):
                if len(track.pos) == 0:
                    continue
                    
                x, y, z = track.pos[:,0], track.pos[:,1], track.pos[:,2]
                t = track.t
                
                # check if this track hit a wireplane
                hit_wireplane = False
                if hasattr(track, 'last_hit_triangles'):
                    hit_wireplane = any(tri_idx < -1 for tri_idx in track.last_hit_triangles)
                    if hit_wireplane:
                        wireplane_hit_indices.append(i)
                        # check if it was also detected
                        if (event.photons_end.flags[i] & SURFACE_DETECT) == SURFACE_DETECT:
                            wireplane_detected_indices.append(i)
                                    
                # get end flag for this photon
                end_flag = event.photons_end.flags[i]
                flag_name = None
                
                # determine flag name for grouping
                for name, flag_value in flag_groups.items():
                    if (end_flag & flag_value) == flag_value:
                        flag_name = name
                        name = f"{flag_name.replace('_', ' ').title()} Track"
                        break
                
                # use default if no match found
                if flag_name is None:
                    flag_name = 'other'
                    name = 'Other Track'
                
                # only add to legend if this is the first track of its type
                show_in_legend = flag_name not in track_traces
                line_color = t
                line_width = 2
                colorscale = track_colorscale
                
                trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(
                        color=line_color,
                        width=line_width,
                        colorscale=colorscale,
                        cmin=min_t,
                        cmax=max_t,
                        colorbar=dict(
                            title="Time (ns)",
                            thickness=20,
                            len=0.75
                        ) if i == 0 and colorscale else None
                    ),
                    marker=dict(size=2),
                    name=name,
                    showlegend=show_in_legend,
                    legendgroup=flag_name,
                    visible=True
                )
                
                fig.add_trace(trace)
                
                # store first trace of each type for button references
                if show_in_legend:
                    track_traces[flag_name] = len(fig.data) - 1
        # Print statistics
        total_photons = len(event.photons_end.flags)
        detected = np.sum(detected_mask)
        no_hit = np.sum(nohit_mask)
        surface_absorbed = np.sum(surface_absorbed_mask)
        bulk_absorbed = np.sum(bulk_absorbed_mask)
        aborted = np.sum((event.photons_end.flags & NAN_ABORT) == NAN_ABORT)
        
        print(f"Total photons: {total_photons}")
        print(f"Detected: {detected} ({detected/total_photons*100:.2f}%)")
        print(f"No Hit: {no_hit} ({no_hit/total_photons*100:.2f}%)")
        print(f"Surface Absorbed: {surface_absorbed} ({surface_absorbed/total_photons*100:.2f}%)")
        print(f"Bulk Absorbed: {bulk_absorbed} ({bulk_absorbed/total_photons*100:.2f}%)")
        print(f"Aborted: {aborted} ({aborted/total_photons*100:.2f}%)")
        
        if 'wireplane_hit_indices' in locals():
            wireplane_hits = len(wireplane_hit_indices)
            print(f"Wireplane Hits: {wireplane_hits} ({wireplane_hits/total_photons*100:.2f}%)")
            if 'wireplane_detected_indices' in locals():
                wireplane_detected = len(wireplane_detected_indices)
                print(f"Wireplane Hits and Detected: {wireplane_detected} ({wireplane_detected/total_photons*100:.2f}%)")
    # Update layout
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=1000,  # increased width
        height=900,  # increased height
        margin=dict(t=100, b=20, l=20, r=20),  # increased top margin for buttons
        legend=dict(
            itemsizing='constant',  # Force constant size for legend items
            itemwidth=30,  # Increase width of legend items
            font=dict(size=12),  # Increase font size
            itemclick="toggle",  # toggle individual traces
            traceorder="normal",
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="rgba(255, 255, 255, 0.5)",
            borderwidth=0
        )
    )
    
    # add buttons to toggle track types if tracks are plotted
    if plot_tracks and hasattr(event, 'photon_tracks') and track_traces:
        buttons = []
        
        # add button for each track type
        for flag_name in track_traces.keys():
            visible_array = [True] * len(fig.data)
            
            # set visibility for tracks of this type
            for i, trace in enumerate(fig.data):
                if hasattr(trace, 'legendgroup') and trace.legendgroup == flag_name:
                    visible_array[i] = True
                elif hasattr(trace, 'legendgroup') and trace.legendgroup in track_traces:
                    visible_array[i] = False
            
            buttons.append(dict(
                label=f"Show only {flag_name.replace('_', ' ').title()} Tracks",
                method="update",
                args=[{"visible": visible_array}]
            ))
        
        # add button to show all tracks
        buttons.append(dict(
            label="Show All Tracks",
            method="update",
            args=[{"visible": [True] * len(fig.data)}]
        ))
        
        # add button to hide all tracks
        hide_tracks = [True] * len(fig.data)
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'legendgroup') and trace.legendgroup in track_traces:
                hide_tracks[i] = False
        
        buttons.append(dict(
            label="Hide All Tracks",
            method="update",
            args=[{"visible": hide_tracks}]
        ))
        
        # add button to show only wireplane hit tracks
        if 'wireplane_hit_indices' in locals() and wireplane_hit_indices:
            wireplane_visible = [True] * len(fig.data)
            track_count = 0
            
            for i, trace in enumerate(fig.data):
                if hasattr(trace, 'legendgroup') and trace.legendgroup in track_traces:
                    # This is a track trace
                    if track_count in wireplane_hit_indices:
                        wireplane_visible[i] = True
                    else:
                        wireplane_visible[i] = False
                    track_count += 1
            
            buttons.append(dict(
                label="Show only Wireplane Hit Tracks",
                method="update",
                args=[{"visible": wireplane_visible}]
            ))
            
            # add button to show only wireplane hit AND detected tracks
            if wireplane_detected_indices:
                wireplane_detected_visible = [True] * len(fig.data)
                track_count = 0
                
                for i, trace in enumerate(fig.data):
                    if hasattr(trace, 'legendgroup') and trace.legendgroup in track_traces:
                        # This is a track trace
                        if track_count in wireplane_detected_indices:
                            wireplane_detected_visible[i] = True
                        else:
                            wireplane_detected_visible[i] = False
                        track_count += 1
                
                buttons.append(dict(
                    label="Show only Wireplane Hit & Detected Tracks",
                    method="update",
                    args=[{"visible": wireplane_detected_visible}]
                ))
                
        # split buttons into two rows with max 5 buttons per row
        row1_buttons = buttons[:5]
        row2_buttons = buttons[5:] if len(buttons) > 5 else []
        
        # add button menus
        button_menus = []
        
        # first row of buttons
        button_menus.append(dict(
            type="buttons",
            direction="right",
            buttons=row1_buttons,
            pad={"r": 5, "t": 5},
            showactive=True,
            x=0.05,
            y=1.05,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            font=dict(size=8)
        ))
        
        # second row of buttons if needed
        if row2_buttons:
            button_menus.append(dict(
                type="buttons",
                direction="right",
                buttons=row2_buttons,
                pad={"r": 5, "t": 5},
                showactive=True,
                x=0.05,
                y=1.0,  # positioned below first row
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                font=dict(size=8)
            ))
        
        fig.update_layout(updatemenus=button_menus)

    fig.show()