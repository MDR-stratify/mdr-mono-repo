# Install dependencies only when needed
FROM node:18-alpine AS deps
WORKDIR /app

# Install optional OS deps if needed (like sharp or node-gyp)
RUN apk add --no-cache libc6-compat

COPY package.json package-lock.json* ./  
RUN npm install

# Build the project
FROM node:18-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN npm run build

# Production image, copy build artifacts
FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV production

# Next.js needs these
RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

USER nextjs

EXPOSE 3000

CMD ["npm", "start"]
