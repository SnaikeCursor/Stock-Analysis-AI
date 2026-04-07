# Stock Analysis AI — Vite build + nginx
# Compose (default): empty VITE_API_BASE_URL + nginx proxies /api → backend
# Split deploy: pass --build-arg VITE_API_BASE_URL=https://your-api.example.com
#               and --build-arg NGINX_CONF=nginx.static.conf

FROM node:20-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

FROM nginx:alpine AS runtime

ARG NGINX_CONF=nginx.compose.conf
COPY --from=build /app/dist /usr/share/nginx/html
COPY ${NGINX_CONF} /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ || exit 1
