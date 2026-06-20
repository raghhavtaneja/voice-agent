-- Schema for the voice-agent app. Run in the Supabase SQL editor.

create extension if not exists "uuid-ossp";

create table if not exists agents (
  id uuid primary key default uuid_generate_v4(),
  name text not null,
  voice_id text not null,
  personality text not null,
  guidelines text not null,
  vapi_phone_number_id text,
  created_at timestamptz not null default now()
);

do $$ begin
  create type document_status as enum ('processing', 'ready', 'failed');
exception when duplicate_object then null;
end $$;

create table if not exists documents (
  id uuid primary key default uuid_generate_v4(),
  agent_id uuid not null references agents(id) on delete cascade,
  filename text not null,
  file_url text not null,
  status document_status not null default 'processing',
  created_at timestamptz not null default now()
);

create index if not exists documents_agent_id_idx on documents(agent_id);

create table if not exists call_logs (
  id uuid primary key default uuid_generate_v4(),
  agent_id uuid not null references agents(id) on delete cascade,
  vapi_call_id text not null,
  transcript jsonb not null default '[]'::jsonb,
  duration_seconds integer,
  created_at timestamptz not null default now()
);

create index if not exists call_logs_agent_id_idx on call_logs(agent_id);
