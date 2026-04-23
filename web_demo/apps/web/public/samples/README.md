# Sample images

Photos shown on the upload screen. Square images look best — the frontend
resizes + center-crops to 224×224 before sending to the model. If a file is
missing, the card falls back to a procedural gradient.

Current samples:

- `cat.jpeg`
- `dog.jpeg`
- `hourse.png`
- `squirrel.png`

## How to change a sample

1. Drop the new image file into this directory (`apps/web/public/samples/`).
   Any format `<img>` can load works (jpg/jpeg/png/webp/gif).
2. Open `apps/web/src/lib/samples.ts` and find the matching entry in
   `MAE_SAMPLES`.
3. Update that entry's `photo` path to point at the new filename (extension
   must match the file on disk), and optionally tweak `name` / `id` /
   `palette` / `accent` to suit.
4. To add a new sample, append a new object to `MAE_SAMPLES`; to remove one,
   delete its entry. The upload grid renders whatever is in the array.
5. Update the "Current samples" list above so this README stays in sync.

No build step needed — Next dev server picks up `public/` changes on reload.
