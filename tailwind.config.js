/**
 * Rebuild the committed stylesheet after changing templates:
 *   tailwindcss -c tailwind.config.js -i callinsights/static/src/input.css \
 *     -o callinsights/static/app.css --minify
 * (Tailwind CLI v3.4, standalone binary from github.com/tailwindlabs/tailwindcss/releases)
 */
module.exports = {
  content: ["./callinsights/templates/**/*.html"],
  theme: { extend: { minWidth: { 10: "2.5rem" } } },
  plugins: [],
};
