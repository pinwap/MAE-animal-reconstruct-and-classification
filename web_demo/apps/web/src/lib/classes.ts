export interface AnimalClassMeta {
  label: string;
  display: string;
}

export const ANIMALS10: AnimalClassMeta[] = [
  { label: "dog",       display: "Dog" },
  { label: "horse",     display: "Horse" },
  { label: "elephant",  display: "Elephant" },
  { label: "butterfly", display: "Butterfly" },
  { label: "chicken",   display: "Chicken" },
  { label: "cat",       display: "Cat" },
  { label: "cow",       display: "Cow" },
  { label: "sheep",     display: "Sheep" },
  { label: "spider",    display: "Spider" },
  { label: "squirrel",  display: "Squirrel" },
];

export function displayName(label: string): string {
  return ANIMALS10.find((c) => c.label === label)?.display ?? label;
}
