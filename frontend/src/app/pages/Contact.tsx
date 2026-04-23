import { FormEvent, useState } from "react";
import { Mail, Phone, Send } from "lucide-react";

import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";

export function Contact() {
  const [submitted, setSubmitted] = useState(false);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitted(true);
    window.setTimeout(() => setSubmitted(false), 2500);
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
        <Card className="rounded-3xl border-border/80 bg-card/80">
          <CardHeader className="gap-4">
            <Badge variant="secondary" className="rounded-full px-3 py-1">
              Contact
            </Badge>
            <CardTitle className="text-4xl font-semibold tracking-tight">
              Keep the support surface simple.
            </CardTitle>
            <CardDescription className="text-base leading-7">
              This page now matches the rest of the interface: quieter layout, default component
              controls, and less custom chrome around the form.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5 text-sm">
            <div className="flex items-center gap-3 rounded-2xl border border-border bg-background px-4 py-4">
              <Mail className="size-4 text-primary" />
              <div>
                <p className="font-medium">Email</p>
                <p className="text-muted-foreground">hello@veritasai.com</p>
              </div>
            </div>
            <div className="flex items-center gap-3 rounded-2xl border border-border bg-background px-4 py-4">
              <Phone className="size-4 text-primary" />
              <div>
                <p className="font-medium">Support line</p>
                <p className="text-muted-foreground">1-800-VERITAS</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-3xl border-border/80 bg-card/80">
          <CardHeader>
            <CardTitle className="text-2xl font-semibold">Send a message</CardTitle>
            <CardDescription>
              Use the same form language and spacing used across the rest of the frontend.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="grid gap-4">
              <div className="grid gap-2">
                <label htmlFor="name" className="text-sm font-medium">
                  Full name
                </label>
                <Input id="name" placeholder="Jane Doe" required />
              </div>
              <div className="grid gap-2">
                <label htmlFor="email" className="text-sm font-medium">
                  Email
                </label>
                <Input id="email" type="email" placeholder="jane@example.com" required />
              </div>
              <div className="grid gap-2">
                <label htmlFor="message" className="text-sm font-medium">
                  Message
                </label>
                <Textarea id="message" placeholder="How can we help?" className="min-h-32" required />
              </div>
              <Button type="submit" className="mt-2 w-fit rounded-full px-6">
                {submitted ? "Sent" : "Send message"}
                <Send className="size-4" />
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
