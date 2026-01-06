# Mondrian Clock Web

A browser-based clock with Mondrian-style design where no two minutes repeat within the 88-billion-year period.

[![Mondrian Clock](screenshot.png)](https://hraj.si/now)

**[▶ Live Demo](https://hraj.si/now)** · **[Wallpaper Engine](https://steamcommunity.com/sharedfiles/filedetails/?id=3125524524)**

## Usage

Just open `index.html` in any browser.

## URL Parameters

- `origin` - Custom epoch (ISO 8601 format, e.g., `2000-01-01T00:00:00Z`)
- `offset` - Year offset from origin
- `period` - Time unit in minutes (1=minutes, 60=hours, 3600=days)
- `mod` - Modulus for display (default: 60)

Example: `https://hraj.si/now?origin=2000-01-01T00:00:00Z`

## Click Interaction

Click the clock to cycle through display modes (all UTC):
- **Seconds** (default): Pattern changes every second within minute
- **Minutes**: Pattern changes every minute within hour
- **Hours**: Pattern changes every hour within day
