/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CHOUCHANE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
