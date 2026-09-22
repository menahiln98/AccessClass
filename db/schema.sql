-- Run this once in the Supabase SQL editor before starting the app.
-- (If you already ran the Portion 1 version of this file, just run the
-- "alter table" and "create table ... revision_queue" statements below —
-- "create table if not exists" for `documents` is a no-op if it already exists.)

create table if not exists documents (
    id uuid primary key default gen_random_uuid(),
    filename text not null,
    storage_path text not null,
    status text not null default 'uploaded',
    parsed_document jsonb,
    accessibility_report jsonb,
    subject_tagged_document jsonb,
    error_message text,
    created_at timestamptz not null default now()
);

-- Portion 2 columns (Stages 4-6).
alter table documents add column if not exists explanation_document jsonb;
alter table documents add column if not exists study_pack jsonb;
alter table documents add column if not exists indexed_chunk_count integer not null default 0;

create index if not exists documents_status_idx on documents (status);

-- Revision queue: sections a student marks as confusing (proposal section 5.4 / Version-One Scope).
create table if not exists revision_queue (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null references documents (id) on delete cascade,
    page_number integer not null,
    note text,
    created_at timestamptz not null default now()
);

create index if not exists revision_queue_document_idx on revision_queue (document_id);
