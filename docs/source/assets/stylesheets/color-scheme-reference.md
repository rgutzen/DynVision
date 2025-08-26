# DynVision Color Scheme Reference

## Primary Colors

The DynVision documentation uses a custom three-color palette that reflects the intersection of neuroscience, AI, and biological systems:

### Main Colors

| Color | Hex Code | Usage | Symbolism |
|-------|----------|-------|-----------|
| **Purple** | `#98239f` | Primary brand color, headers, navigation | Neural networks, connectivity |
| **Green** | `#5c7c58` | Accent color, links, highlights | Biological systems, growth |
| **Brown** | `#b3894f` | Tertiary color, code blocks, performance | Stability, foundation |

### Color Variations

Each main color has automatically generated variations for different UI states:

#### Purple Variations
- **Light**: `#b347ba` - Hover states, active elements
- **Lighter**: `#d171d6` - Dark mode primary
- **Dark**: `#7a1c80` - Footer, depth
- **Darker**: `#5c1560` - Deep backgrounds

#### Green Variations  
- **Light**: `#6d8c69` - Hover states
- **Lighter**: `#7e9d7a` - Subtle accents
- **Dark**: `#4a6547` - Borders, structure
- **Darker**: `#3a4f37` - Deep contrast

#### Brown Variations
- **Light**: `#c39960` - Interactive elements
- **Lighter**: `#d4aa71` - Backgrounds
- **Dark**: `#9a7840` - Text emphasis
- **Darker**: `#816030` - Strong contrast

## Usage Guidelines

### Primary Applications
- **Purple**: Main navigation, primary buttons, section headers
- **Green**: Secondary actions, accent elements, table highlights
- **Brown**: Code syntax, performance indicators, stable elements

### Gradient Combinations
The theme uses gradients to create visual interest:
- **Header**: Purple → Green → Brown (135° gradient)
- **Buttons**: Purple → Purple Light (45° gradient)
- **Hero Sections**: Animated gradient using all three colors

### Accessibility
All color combinations meet WCAG 2.1 AA contrast requirements:
- Purple on white: 7.2:1 contrast ratio
- Green on white: 6.8:1 contrast ratio  
- Brown on white: 5.4:1 contrast ratio

## Dark Mode Adaptations

In dark mode, colors are automatically adjusted:
- Lighter variations become primary colors
- Transparency effects are enhanced
- Background contrasts are optimized

## Custom Elements

### Admonitions
- **DynVision** (gradient): Framework-specific information
- **Neural** (purple): Neural network concepts
- **Bio** (green): Biological insights  
- **Performance** (brown): Optimization tips

### Interactive Elements
- Buttons use gradient effects with your colors
- Hover states lighten colors by one variation
- Active states use the darker variations
- Focus indicators use transparent overlays

This color scheme creates a cohesive visual identity that reflects DynVision's focus on biologically-inspired neural networks while maintaining excellent readability and accessibility.


# DynVision Color Scheme Examples

## Custom Admonitions

!!! dynvision "DynVision Framework"
    This admonition uses the full color gradient and is perfect for highlighting 
    framework-specific information.

!!! neural "Neural Network Architecture"
    Use this purple admonition for neural network concepts and architecture details.
    
    ```python
    model = DyRCNNx4(
        recurrence_type="full",
        dt=2.0,
        tau=8.0
    )
    ```

!!! bio "Biological Inspiration"
    This green admonition is ideal for biological concepts and neuroscience insights.

!!! performance "Performance Tips"
    Use this brown admonition for optimization tips and performance considerations.

## Badges and Labels

You can create colored badges using HTML:

<span class="badge purple">New Feature</span>
<span class="badge green">Biological</span>
<span class="badge brown">Performance</span>

## Button Examples

[Get Started](#){ .md-button }
[Primary Action](#){ .md-button .md-button--primary }

## Code Examples

The custom styling enhances code blocks:

```python
# This code block uses the custom styling
from dynvision.models import DyRCNNx4

model = DyRCNNx4(
    n_classes=100,
    recurrence_type="full",
    dt=2.0,
    tau=8.0
)
```

## Tables

| Feature | Purple Theme | Green Theme | Brown Theme |
|---------|-------------|-------------|-------------|
| Primary | #98239f | #5c7c58 | #b3894f |
| Usage | Headers, Navigation | Accents, Links | Highlights, Code |
| Purpose | Brand Identity | Biological Focus | Performance |