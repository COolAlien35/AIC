---
version: alpha
name: AIC War Room — Emerald Obsidian
description: Dark, glassmorphic command-center design system for an AI incident-response product. The visual identity pairs warm obsidian surfaces, emerald-led status signaling, disciplined typography, and restrained motion so dense operational data feels premium, calm, and legible under pressure.
colors:
  background: "#0C0F0A"
  surface: "#111610"
  surface-elevated: "#161B14"
  surface-card: "#161E14"
  surface-contrast: "#1C2618"
  grid: "#1E2B1A"
  text-primary: "#E8EDE6"
  text-secondary: "#9CA89A"
  text-muted: "#6B7A68"
  text-subtle: "#3D4A3B"
  primary: "#10B981"
  primary-strong: "#059669"
  primary-bright: "#34D399"
  primary-soft: "#6EE7B7"
  secondary: "#14B8A6"
  secondary-soft: "#2DD4BF"
  tertiary: "#A78BFA"
  warning: "#FBBF24"
  warning-strong: "#F59E0B"
  danger: "#FB7185"
  danger-strong: "#F43F5E"
  danger-deep: "#E11D48"
  neutral-300: "#CBD5E1"
  neutral-400: "#94A3B8"
  neutral-500: "#64748B"
  neutral-600: "#475569"
  on-background: "#E8EDE6"
  on-surface: "#E8EDE6"
  on-primary: "#0C0F0A"
surfaces:
  glass: "rgba(16, 24, 14, 0.45)"
  glass-hover: "rgba(22, 32, 18, 0.60)"
  card: "rgba(22, 30, 20, 0.65)"
  card-hover: "rgba(28, 38, 24, 0.80)"
  panel-soft: "rgba(22, 30, 20, 0.50)"
  overlay-positive-subtle: "rgba(52, 211, 153, 0.04)"
  overlay-positive-soft: "rgba(52, 211, 153, 0.06)"
  overlay-positive-strong: "rgba(52, 211, 153, 0.12)"
  overlay-warning-soft: "rgba(251, 191, 36, 0.06)"
  overlay-warning-strong: "rgba(251, 191, 36, 0.12)"
  overlay-danger-soft: "rgba(251, 113, 133, 0.06)"
  overlay-danger-strong: "rgba(251, 113, 133, 0.10)"
  overlay-info-soft: "rgba(167, 139, 250, 0.12)"
  chart-danger-fill: "rgba(251, 113, 133, 0.08)"
gradients:
  brand: "linear-gradient(135deg, #059669, #10B981, #34D399)"
  brand-hero: "linear-gradient(135deg, #059669, #10B981, #34D399, #6EE7B7)"
  glass: "linear-gradient(135deg, rgba(52, 211, 153, 0.06), rgba(20, 184, 166, 0.03))"
  business-alert: "linear-gradient(135deg, rgba(251, 191, 36, 0.06), rgba(251, 113, 133, 0.04))"
  postmortem-header: "linear-gradient(135deg, rgba(52, 211, 153, 0.06), rgba(20, 184, 166, 0.03))"
  sidebar-ambient: "radial-gradient(ellipse at 50% 0%, rgba(5, 150, 105, 0.06) 0%, transparent 60%)"
  canvas-left: "radial-gradient(ellipse at 20% 50%, rgba(5, 150, 105, 0.04) 0%, transparent 60%)"
  canvas-top-right: "radial-gradient(ellipse at 80% 20%, rgba(20, 184, 166, 0.03) 0%, transparent 50%)"
  divider: "linear-gradient(90deg, transparent, rgba(52, 211, 153, 0.15), transparent)"
  footer-rule: "linear-gradient(90deg, transparent, rgba(52, 211, 153, 0.30), transparent)"
borders:
  subtle: "rgba(52, 211, 153, 0.08)"
  default: "rgba(52, 211, 153, 0.15)"
  hover: "rgba(52, 211, 153, 0.30)"
  active: "rgba(52, 211, 153, 0.50)"
  root-cause: "#FBBF24"
  danger: "#F43F5E"
  chart-cursor: "rgba(255, 255, 255, 0.30)"
typography:
  display-hero:
    fontFamily: Outfit
    fontSize: 38px
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.03em"
  headline-lg:
    fontFamily: Outfit
    fontSize: 32px
    fontWeight: 600
    lineHeight: 40px
    letterSpacing: "-0.02em"
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: 600
    lineHeight: 32px
    letterSpacing: "-0.015em"
  title-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: 600
    lineHeight: 24px
    letterSpacing: "0.01em"
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.6
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.6
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.5
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: 600
    lineHeight: 16px
    letterSpacing: "0.08em"
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: 500
    lineHeight: 16px
    letterSpacing: "0.08em"
  numeric-lg:
    fontFamily: JetBrains Mono
    fontSize: 32px
    fontWeight: 700
    lineHeight: 1.1
  numeric-md:
    fontFamily: JetBrains Mono
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.2
  caption:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: 400
    lineHeight: 18px
rounded:
  xs: 6px
  sm: 10px
  md: 12px
  lg: 14px
  xl: 16px
  pill: 20px
  full: 9999px
spacing:
  micro: 4px
  tight: 6px
  xs: 8px
  sm: 10px
  md: 12px
  lg: 14px
  xl: 16px
  xxl: 18px
  section-gap: 24px
  page-margin: 24px
  card-padding: 16px
  card-padding-roomy: 18px
  hero-padding: 20px
  panel-gap: 16px
  section-margin: 40px
motion:
  duration-fast: "150ms"
  duration-normal: "250ms"
  duration-slow: "400ms"
  duration-pulse: "2000ms"
  duration-ambient: "3000ms"
  easing-smooth: "cubic-bezier(0.4, 0, 0.2, 1)"
  easing-out: "cubic-bezier(0, 0, 0.2, 1)"
  easing-spring: "cubic-bezier(0.34, 1.56, 0.64, 1)"
  entrance-offset-y: 12px
  entrance-offset-x: 16px
backdrop:
  glass-card: "blur(16px) saturate(140%)"
  glass-panel: "blur(12px)"
  glass-soft: "blur(8px)"
shadows:
  card: "0 4px 24px rgba(0, 0, 0, 0.40), 0 1px 3px rgba(0, 0, 0, 0.30)"
  card-hover: "0 8px 40px rgba(0, 0, 0, 0.50), 0 0 20px rgba(52, 211, 153, 0.08)"
  glow-primary: "0 0 30px rgba(52, 211, 153, 0.12), 0 0 60px rgba(52, 211, 153, 0.05)"
  glow-warning: "0 0 20px rgba(245, 158, 11, 0.12)"
  glow-danger: "0 0 20px rgba(244, 63, 94, 0.12)"
elevation:
  canvas:
    backgroundColor: "{colors.background}"
    backgroundImage: "{gradients.canvas-left}, {gradients.canvas-top-right}"
  glass-card:
    backgroundColor: "{surfaces.glass}"
    borderColor: "{borders.default}"
    backdropFilter: "{backdrop.glass-card}"
    shadow: "{shadows.card}"
  glass-card-hover:
    backgroundColor: "{surfaces.glass-hover}"
    borderColor: "{borders.hover}"
    shadow: "{shadows.card-hover}"
  soft-panel:
    backgroundColor: "{surfaces.panel-soft}"
    borderColor: "{borders.subtle}"
    backdropFilter: "{backdrop.glass-soft}"
  sidebar:
    backgroundColor: "{colors.surface}"
    backgroundImage: "{gradients.sidebar-ambient}"
    borderColor: "{borders.subtle}"
visualization:
  plot-background: "{colors.background}"
  plot-grid: "{colors.grid}"
  plot-text: "{colors.text-secondary}"
  threshold-line: "{colors.warning}"
  current-step-line: "{borders.chart-cursor}"
  series-trained: "{colors.primary-bright}"
  series-untrained: "{colors.secondary}"
  series-baseline: "{colors.warning}"
  series-risk: "{colors.danger}"
  series-neutral: "{colors.neutral-500}"
components:
  live-status-pill:
    backgroundColor: "rgba(52, 211, 153, 0.08)"
    textColor: "{colors.primary-bright}"
    typography: "{typography.label-md}"
    rounded: "{rounded.pill}"
    padding: "4px 16px"
  hero-title:
    textColor: "{colors.primary-bright}"
    typography: "{typography.display-hero}"
  metric-card:
    backgroundColor: "rgba(16, 24, 14, 0.45)"
    textColor: "{colors.text-primary}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.lg}"
    padding: 18px
  metric-card-hover:
    backgroundColor: "rgba(22, 32, 18, 0.60)"
  metric-label:
    textColor: "{colors.text-muted}"
    typography: "{typography.label-sm}"
  metric-value:
    textColor: "{colors.text-primary}"
    typography: "{typography.numeric-md}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    typography: "{typography.title-md}"
    rounded: "{rounded.sm}"
    padding: "0.55rem 1.6rem"
  button-primary-hover:
    backgroundColor: "{colors.primary-bright}"
  tab:
    backgroundColor: "rgba(52, 211, 153, 0.06)"
    textColor: "{colors.text-secondary}"
    typography: "{typography.body-md}"
    rounded: "{rounded.sm}"
    padding: "8px 16px"
  tab-active:
    backgroundColor: "rgba(52, 211, 153, 0.12)"
    textColor: "{colors.primary-soft}"
    typography: "{typography.title-md}"
    rounded: "{rounded.sm}"
  sidebar-panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.lg}"
  command-brief:
    backgroundColor: "rgba(52, 211, 153, 0.04)"
    textColor: "{colors.warning}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.md}"
    padding: "10px 14px"
  agent-card:
    backgroundColor: "rgba(22, 30, 20, 0.65)"
    textColor: "{colors.text-primary}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.md}"
    padding: "10px 14px"
  agent-card-adversary:
    backgroundColor: "rgba(251, 113, 133, 0.04)"
    textColor: "{colors.danger}"
    rounded: "{rounded.md}"
  leaderboard-row-aic:
    backgroundColor: "rgba(52, 211, 153, 0.12)"
    textColor: "{colors.primary-bright}"
    typography: "{typography.body-md}"
    rounded: "{rounded.md}"
    padding: "14px 18px"
  leaderboard-row-baseline:
    backgroundColor: "rgba(107, 122, 104, 0.04)"
    textColor: "{colors.text-secondary}"
    typography: "{typography.body-md}"
    rounded: "{rounded.md}"
    padding: "14px 18px"
  postmortem-panel:
    backgroundColor: "rgba(22, 30, 20, 0.50)"
    textColor: "#B8C4B6"
    typography: "{typography.body-md}"
    rounded: "{rounded.md}"
    padding: 16px
  debate-veto:
    backgroundColor: "rgba(251, 113, 133, 0.06)"
    textColor: "{colors.danger}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.md}"
    padding: "10px 14px"
  debate-warning:
    backgroundColor: "rgba(251, 191, 36, 0.06)"
    textColor: "{colors.warning}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.md}"
    padding: "10px 14px"
  debate-support:
    backgroundColor: "rgba(52, 211, 153, 0.06)"
    textColor: "{colors.primary-bright}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.md}"
    padding: "10px 14px"
---

## Overview

This design system expresses a high-stakes autonomous incident command center. It should feel like a premium war room rather than a generic admin console: dark, focused, and calm under pressure, with just enough glow and motion to suggest live telemetry without becoming noisy.

The brand personality is **strategic, technical, and quietly confident**. The interface is data-dense, but never frantic. Warm obsidian surfaces keep the product grounded and serious, while emerald becomes the dominant signal for trust, recovery, selection, and forward momentum. The result is a dashboard that reads like an elite operations cockpit: sophisticated, legible, and emotionally controlled even when the underlying scenario is unstable.

## Colors

The palette is anchored in a warm charcoal-black foundation rather than a cold navy. That choice matters: it makes the experience feel tactile, premium, and less sci-fi, while still supporting strong contrast for charts, metrics, and alerts.

- **Obsidian backgrounds** establish the canvas. Surfaces step from `background` to `surface`, `surface-elevated`, and `surface-card` to create separation without relying on bright panels.
- **Emerald is the hero accent**. It drives primary actions, success states, active tabs, health-positive indicators, progress fills, and branded gradient moments.
- **Teal and jade act as secondary technical accents**. They support comparison states, topology visuals, and auxiliary emphasis without competing with emerald.
- **Gold communicates caution and operational time pressure**. Use it for thresholds, warnings, commander strategy emphasis, and “watch closely” signals.
- **Rose communicates risk, adversarial behavior, unsafe paths, and failure pressure**. It should feel urgent and specific, not broadly decorative.
- **Muted gray-greens and soft slates carry metadata**. Secondary text must remain readable but recessive, especially around dense visualizations and telemetry labels.

Transparent overlays are part of the color language. Positive, warning, and danger states often appear as lightly tinted glass fills with strong colored edge accents instead of fully filled blocks.

## Typography

Typography splits into three functional roles:

- **Outfit** is reserved for hero-level identity moments and section headers. It gives the product a modern, composed, slightly futuristic voice without feeling decorative.
- **Inter** is the workhorse UI face. It should be used for operational copy, component labels, tab text, annotations, and explanatory prose.
- **JetBrains Mono** is the telemetry face. It is used for numbers, deltas, trust scores, compact metadata, slider values, and technical labeling wherever precision is part of the message.

The hierarchy should emphasize clarity over flourish. Large branded headlines can be expressive, but most of the interface should stay efficient and compact. Monospace labels are frequently uppercase with generous tracking to make them read like instrumentation rather than marketing copy.

## Layout

The layout is a **wide control-room dashboard** built from stacked panels, modular cards, and tabbed workspaces. It should feel intentionally segmented so operators can scan quickly across status, reasoning, and outcomes.

- Use a consistent small-to-medium spacing rhythm derived from the 8px family, with common working values between 8px and 18px.
- Prefer dense but breathable panel groupings. Cards and glass containers should usually carry 14–18px internal padding.
- Horizontal comparison is important. Many views work best in side-by-side columns so the user can evaluate system state, trust behavior, and outcomes simultaneously.
- Top-level navigation should feel like a set of mission modes rather than casual tabs.
- The sidebar should act as a control rail: compact, always available, and visually subordinate to the main operational canvas.

Whitespace is used strategically, not generously. The product is meant to show a lot of information, but it must still preserve visual hierarchy through grouping, contrast, and consistent card spacing.

## Elevation & Depth

Depth comes from **glassmorphism plus restrained glow**, not from heavy layered shadows. Most major containers are semi-transparent dark panels with blur and subtle saturation, sitting above the obsidian canvas.

- Standard operational cards use translucent green-tinted glass with a thin emerald border and soft black shadow.
- Hover states brighten the surface slightly, strengthen the border, and introduce a wider, softer emerald glow.
- Critical or adversarial states swap emerald glow for rose glow.
- Soft blur values between 8px and 16px maintain the “instrument panel under glass” effect.
- Visualizations should blend into the background instead of looking like separate white chart widgets.

This system should always feel lightweight and luminous. Even when the interface is dense, the panels should appear to float rather than stack like heavy slabs.

## Shapes

The shape language is **soft-technical**. Corners are rounded enough to feel modern and premium, but not so rounded that the interface becomes playful.

- Standard controls and inputs live around the 10px radius.
- Cards, tabs, and panel containers live around the 12–14px range.
- Hero containers can stretch to 16px.
- Pills, badges, and progress capsules use extreme radius values for a clean, instrument-like silhouette.
- Vertical accent rules at 3px are used to mark importance, urgency, or provenance in alerts and recommendation blocks.

Use circular forms sparingly and meaningfully, primarily for topology nodes, point markers, and compact status icons.

## Components

### Metric Cards

Metric cards are the foundational atom of the system. They use dark glass, mono labels, mono numeric values, and soft emerald borders. On hover, they should lift slightly and glow, but never become flashy.

### Buttons

Primary buttons use the emerald gradient and dark-on-bright text treatment. They should feel energetic and decisive. Hover states lift and brighten, while pressed states snap back quickly to reinforce responsiveness.

### Tabs

Tabs are compact glass pills that shift from muted metadata styling to soft emerald emphasis when selected. The active state should read as “this mission mode is live.”

### Recommendation and Debate Cards

Agent and debate cards rely on left-edge accent bars and tinted glass fills to communicate intent. Green signals support or healthy recommendations, gold signals caution or challenge, and rose signals vetoes or adversarial risk. These cards should feel analytical, not conversational.

### Leaderboards and Comparison Panels

Comparison surfaces should keep the same dark-glass language but allow stronger semantic highlighting. The primary system entry may receive a richer emerald tint, while baselines remain subdued and neutral.

### Charts, Gauges, and Topology Visuals

Charts should preserve the same palette logic as the rest of the system: emerald for trained or healthy behavior, teal for alternate paths, gold for thresholds, rose for danger, and muted gray for neutral comparison. Gridlines must stay subtle. Labels should remain low-contrast but readable.

### Postmortem and Narrative Panels

Narrative blocks keep the same visual family but soften the contrast slightly so longer passages are easier to read. These panels should feel like calm briefing documents generated from the live operational view.

## Do's and Don'ts

- **Do** keep the overall mood dark, warm, and premium.
- **Do** use emerald as the default signal for action, recovery, trust, and product identity.
- **Do** pair dense telemetry with monospace labels and values so data feels precise.
- **Do** preserve translucent surfaces, blur, and soft glows instead of flattening everything into solid blocks.
- **Do** reserve gold for warning or strategic attention, and rose for real risk or adversarial pressure.
- **Don't** introduce large bright blue surfaces as the dominant brand layer.
- **Don't** use heavy opaque cards that break the glass command-center feel.
- **Don't** over-animate. Motion should confirm liveness and hierarchy, not distract from operations.
- **Don't** mix sharp-cornered components into the main experience.
- **Don't** use rose as a decorative accent; it should remain meaningfully tied to danger, failure, or sabotage.